#!/usr/bin/env python3
"""Linearised BCS gap equation driven by the SU(2) gauge-boson propagator.

For each input CSV (one per beta value) this script:

  1. Extracts the static gauge-boson propagator G(r) from Gprop_r* columns.
  2. Fits log G(r) vs r to a Yukawa form, obtaining (m, Z).
  3. Builds the linearised BCS gap equation
            Delta(k) = -sum_{k'} V_eff(k-k') * F(k', T) * Delta(k')
     with  V_eff(q) = -g^2 * Z / (qhat^2 + m^2)
            qhat^2  = 4 sum_i sin^2((k_i-k_i')/2)        (lattice momentum)
            F(k',T) = tanh(|xi(k')|/(2T)) / (2|xi(k')|)
            xi(k)   = eps(k) - mu
            eps(k)  = -2 t (cos kx + cos ky + cos kz)  (3D simple cubic)
  4. Finds T_c where the largest eigenvalue of the kernel matrix = 1.

Produces:
  - A table of (beta, m, Z, T_c)
  - A plot of T_c vs beta with the deconfinement reference line shown.

Usage:
    python scripts/gap_equation.py path1.csv path2.csv ...
                                   [--g2 G2] [--mu MU] [--N_grid N]
                                   [--out PATH]

The free parameter `g2` is the boson-electron coupling.  T_c at any
other g2' follows from the linearised gap equation: scaling g2 -> g2'
multiplies the kernel everywhere, and the bisection is monotone in
g2.  We default to g2=1 for a reference scale.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from scipy.sparse.linalg import eigs as _scipy_eigs
except ImportError:
    _scipy_eigs = None


# --------------------------------------------------------------------------
# CSV parsing and propagator fit
# --------------------------------------------------------------------------

def parse_header(path):
    cfg = {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            for m in re.finditer(r"(\w+)=([\w.+\-]+)", line):
                cfg[m.group(1)] = m.group(2)
    return cfg


def load_propagator(csv_path):
    cfg = parse_header(csv_path)
    df  = pd.read_csv(csv_path, comment="#")
    if cfg.get("propagator", "0") != "1":
        return None
    r_max = int(cfg.get("r_max", 0))
    cols = [f"Gprop_r{r}" for r in range(r_max + 1)
            if f"Gprop_r{r}" in df.columns]
    if not cols:
        return None
    G = df[cols].dropna().values
    if G.size == 0:
        return None
    G_mean = G.mean(axis=0)
    # Per-r statistical error of the mean.
    G_err  = (G.std(axis=0, ddof=1) / np.sqrt(G.shape[0])
              if G.shape[0] > 1 else np.zeros_like(G_mean))
    return cfg, G_mean, G_err


def fit_yukawa(G_mean, G_err=None, r_lo=1, r_hi=None,
               sn_threshold=3.0):
    """Fit log G(r) = -m r + log Z over r in [r_lo, r_hi].

    If G_err is provided, the fit is error-weighted (so noise-dominated
    points are downweighted), and r values where G(r) < sn_threshold *
    G_err(r) are excluded entirely.  This is essential at low beta where
    G(r) drops below the noise floor by r = 3 or so."""
    rs = np.arange(len(G_mean))
    if r_hi is None:
        r_hi = max(2, len(G_mean) // 2 - 1)
    mask = (rs >= r_lo) & (rs <= r_hi) & (G_mean > 0)
    # S/N filter: drop r values whose mean is comparable to its error.
    if G_err is not None and np.any(G_err > 0):
        sn_ok = G_mean > sn_threshold * G_err
        mask = mask & sn_ok
    if mask.sum() < 2:
        return None, None

    x = rs[mask].astype(float)
    y = np.log(G_mean[mask])

    if G_err is not None and np.any(G_err[mask] > 0):
        # Error-weighted least squares.  In log space, sigma_y ~ sigma_G / G.
        sigma_y = G_err[mask] / G_mean[mask]
        sigma_y = np.where(sigma_y > 0, sigma_y, 1e-6)
        w = 1.0 / sigma_y ** 2
        sw   = w.sum()
        swx  = (w * x).sum()
        swy  = (w * y).sum()
        swxx = (w * x * x).sum()
        swxy = (w * x * y).sum()
        denom = sw * swxx - swx ** 2
        if abs(denom) < 1e-30:
            return None, None
        slope     = (sw * swxy - swx * swy) / denom
        intercept = (swy - slope * swx) / sw
    else:
        coef = np.polyfit(x, y, 1)
        slope, intercept = coef[0], coef[1]
    m, logZ = -slope, intercept
    return float(m), float(np.exp(logZ))


# --------------------------------------------------------------------------
# BCS kernel & eigenvalue search
# --------------------------------------------------------------------------

def lattice_dispersion(N_grid, mu, t=1.0):
    """Return KX, KY, KZ on a centred BZ grid and xi = eps - mu."""
    n  = np.arange(N_grid)
    ks = (n - N_grid // 2) * (2.0 * np.pi / N_grid)
    KX, KY, KZ = np.meshgrid(ks, ks, ks, indexing="ij")
    eps = -2.0 * t * (np.cos(KX) + np.cos(KY) + np.cos(KZ))
    xi  = eps - mu
    return KX, KY, KZ, xi


def build_kernel(N_grid, m, Z, g2, mu, T,
                 t=1.0, eps_min=1e-6):
    """Build the linearised BCS kernel matrix on the N_grid^3 momentum mesh."""
    KX, KY, KZ, xi = lattice_dispersion(N_grid, mu, t)

    # Flat list of momenta
    Kf = np.column_stack([KX.flatten(), KY.flatten(), KZ.flatten()])
    xi_flat = xi.flatten()
    abs_xi  = np.maximum(np.abs(xi_flat), eps_min)

    F = np.tanh(abs_xi / (2.0 * T)) / (2.0 * abs_xi)        # length N

    # Periodic lattice momentum-squared:  qhat^2 = 4 sum_i sin^2(q_i/2)
    diff = Kf[:, None, :] - Kf[None, :, :]                  # (N, N, 3)
    qhat2 = (4.0 * np.sin(diff / 2.0) ** 2).sum(axis=-1)    # (N, N)

    # V(q) = -g^2 Z / (qhat^2 + m^2)
    V = -g2 * Z / (qhat2 + m * m)                           # (N, N)

    # K(k, k') = -V(k-k') * F(k')
    K = -V * F[None, :]                                     # (N, N)
    return K


def V_eff_from_data(qhat2, G_mean, g2):
    """Convert lattice-measured G(r) directly into V_eff(qhat^2) via cosine
    transform.  G(0) is intentionally EXCLUDED -- it is the equal-position
    autocorrelator <|A(x)|^2>, a UV-cutoff-dependent local quantity that
    is not part of the Cooper-channel kernel.  Only r >= 1 contributes,
    which is the long-range (non-local) propagator we actually want.
    No Yukawa assumption -- robust to non-Yukawa structure (nodes etc.)."""
    q_eff = np.sqrt(np.clip(qhat2, 0.0, None))
    V = np.zeros_like(q_eff)
    for r in range(1, len(G_mean)):
        V = V + 2.0 * G_mean[r] * np.cos(q_eff * r)
    return -g2 * V


def build_kernel_from_data(N_grid, G_mean, g2, mu, T,
                            t=1.0, eps_min=1e-6):
    """BCS kernel using V_eff sourced from G(r) data, no fit."""
    KX, KY, KZ, xi = lattice_dispersion(N_grid, mu, t)
    Kf = np.column_stack([KX.flatten(), KY.flatten(), KZ.flatten()])
    abs_xi = np.maximum(np.abs(xi.flatten()), eps_min)
    F = np.tanh(abs_xi / (2.0 * T)) / (2.0 * abs_xi)
    diff  = Kf[:, None, :] - Kf[None, :, :]
    qhat2 = (4.0 * np.sin(diff / 2.0) ** 2).sum(axis=-1)
    V = V_eff_from_data(qhat2, G_mean, g2)
    return -V * F[None, :]


def find_Tc_from_data(N_grid, G_mean, g2, mu,
                       T_lo=1e-3, T_hi=50.0, tol=1e-3, max_bisect=40):
    """Bisect T_c using the V_eff-from-data kernel."""
    def lam_at(T):
        return lambda_max(build_kernel_from_data(N_grid, G_mean, g2, mu, T))
    lam_lo = lam_at(T_lo)
    lam_hi = lam_at(T_hi)
    if lam_lo < 1.0:
        return None, ("no instability: lambda(T_lo)<1, T_lo=" + f"{T_lo:.3g}")
    expansion = 0
    while lam_hi > 1.0 and expansion < 4:
        T_hi *= 2.0
        lam_hi = lam_at(T_hi)
        expansion += 1
    if lam_hi > 1.0:
        return None, ("always unstable up to T_hi=" + f"{T_hi:.3g}")
    for _ in range(max_bisect):
        T_mid = 0.5 * (T_lo + T_hi)
        lam   = lam_at(T_mid)
        if lam > 1.0:
            T_lo = T_mid
        else:
            T_hi = T_mid
        if (T_hi - T_lo) / max(T_hi, 1e-12) < tol:
            break
    return 0.5 * (T_lo + T_hi), None


def lambda_max(K):
    """Largest absolute eigenvalue of K via Arnoldi (one Lanczos-like step)."""
    # K can be moderately dense; for N=8 grid -> 512x512 -> dense numpy is fine.
    if K.shape[0] <= 1024:
        evals = np.linalg.eigvals(K)
        return float(np.max(np.abs(evals)))
    # Larger: sparse iterative
    if _scipy_eigs is None:
        evals = np.linalg.eigvals(K)
        return float(np.max(np.abs(evals)))
    vals, _ = _scipy_eigs(K, k=1, which="LM", maxiter=2000, tol=1e-6)
    return float(np.max(np.abs(vals)))


def find_Tc(N_grid, m, Z, g2, mu,
            T_lo=1e-3, T_hi=2.0, tol=1e-3, max_bisect=40):
    """Bisect on T to find lambda_max(K(T)) = 1.  Returns None if no
    crossing inside [T_lo, T_hi]."""
    lam_lo = lambda_max(build_kernel(N_grid, m, Z, g2, mu, T_lo))
    lam_hi = lambda_max(build_kernel(N_grid, m, Z, g2, mu, T_hi))
    if lam_lo < 1.0:
        return None, ("no instability: lambda(T_lo)<1, T_lo=" + f"{T_lo:.3g}")
    # Auto-widen if "always unstable" -- try doubling T_hi up to 16x.
    expansion = 0
    while lam_hi > 1.0 and expansion < 4:
        T_hi *= 2.0
        lam_hi = lambda_max(build_kernel(N_grid, m, Z, g2, mu, T_hi))
        expansion += 1
    if lam_hi > 1.0:
        return None, ("always unstable: lambda(T_hi)>1 even at T_hi="
                      f"{T_hi:.3g} (auto-widened {expansion} times); "
                      "rerun with larger --T_hi or smaller --g2")

    for _ in range(max_bisect):
        T_mid = 0.5 * (T_lo + T_hi)
        lam   = lambda_max(build_kernel(N_grid, m, Z, g2, mu, T_mid))
        if lam > 1.0:
            T_lo = T_mid
        else:
            T_hi = T_mid
        if (T_hi - T_lo) / max(T_hi, 1e-12) < tol:
            break
    return 0.5 * (T_lo + T_hi), None


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("csvs", nargs="+", type=Path)
    p.add_argument("--g2",     type=float, default=1.0,
                   help="Electron-boson coupling g^2 (default 1.0)")
    p.add_argument("--mu",     type=float, default=0.0,
                   help="Chemical potential (default 0 = half-filling)")
    p.add_argument("--N_grid", type=int, default=8,
                   help="Momentum grid size per spatial axis (default 8)")
    p.add_argument("--T_hi",   type=float, default=50.0)
    p.add_argument("--T_lo",   type=float, default=1e-3)
    p.add_argument("--use_data", action="store_true",
                   help="Bypass Yukawa fit; use measured G(r) directly "
                        "in the BCS kernel via cosine FT.  Robust to "
                        "non-Yukawa propagator structure (nodes etc.).")
    p.add_argument("--r_hi",   type=int, default=None,
                   help="Yukawa fit upper r-bound; default = Lx/2 - 1.  "
                        "Set to 2 for very heavy bosons (low beta) where "
                        "late-r points are noise-dominated.")
    p.add_argument("--out",    type=Path, default=Path("Tc_vs_beta.png"))
    p.add_argument("--beta_c", type=float, default=2.298,
                   help="Reference deconfinement beta_c on Nt=4 (literature)")
    args = p.parse_args()

    rows = []
    for csv in sorted(args.csvs):
        loaded = load_propagator(csv)
        if loaded is None:
            print(f"  skipping {csv.name}: no propagator data", file=sys.stderr)
            continue
        cfg, G_mean, G_err = loaded
        beta = float(cfg.get("beta", 0))
        Lx   = int(cfg.get("Lx", 0))
        if args.use_data:
            Tc, msg = find_Tc_from_data(args.N_grid, G_mean, args.g2,
                                          args.mu,
                                          T_lo=args.T_lo, T_hi=args.T_hi)
            m_print, Z_print = float("nan"), float("nan")
        else:
            m, Z = fit_yukawa(G_mean, G_err, r_hi=args.r_hi)
            if m is None or m <= 0:
                print(f"  skipping {csv.name}: bad Yukawa fit",
                      file=sys.stderr)
                continue
            Tc, msg = find_Tc(args.N_grid, m, Z, args.g2, args.mu,
                               T_lo=args.T_lo, T_hi=args.T_hi)
            m_print, Z_print = m, Z
        rows.append({"beta": beta, "m": m_print, "Z": Z_print,
                      "Tc": Tc, "note": msg or ""})
        Tc_str = f"{Tc:.4f}" if Tc is not None else f"-- ({msg})"
        print(f"  beta={beta:.3f}: m={m_print:.4f}, Z={Z_print:.4f}, "
              f"Tc={Tc_str}")

    if not rows:
        sys.exit("No valid CSVs found.")

    df = (pd.DataFrame(rows).sort_values("beta").reset_index(drop=True))
    df.to_csv(args.out.with_suffix(".csv"), index=False)
    print(f"\nWrote table  -> {args.out.with_suffix('.csv')}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1.  m(beta)
    ax = axes[0]
    ax.plot(df["beta"], df["m"], "o-")
    ax.axvline(args.beta_c, color="gray", linestyle=":",
               label=fr"$\beta_c \approx {args.beta_c:.3f}$")
    ax.set_xlabel(r"$\beta$"); ax.set_ylabel(r"screening mass $m$")
    ax.set_title("Boson screening mass vs $\\beta$")
    ax.legend(); ax.grid(alpha=0.3)

    # 2.  Z(beta)
    ax = axes[1]
    ax.plot(df["beta"], df["Z"], "o-")
    ax.axvline(args.beta_c, color="gray", linestyle=":")
    ax.set_xlabel(r"$\beta$"); ax.set_ylabel(r"propagator amplitude $Z$")
    ax.set_title("Yukawa amplitude vs $\\beta$")
    ax.grid(alpha=0.3)

    # 3.  T_c(beta)
    ax = axes[2]
    found = df["Tc"].notna()
    if found.any():
        ax.plot(df.loc[found, "beta"], df.loc[found, "Tc"],
                "o-", color="tab:red", label="BCS $T_c$")
    not_found = ~found
    if not_found.any():
        # Plot "no result" markers at y=0 but make the message unambiguous.
        # If the script can't find T_c, this is bisection-window failure,
        # NOT a physical zero.  Bump --T_hi or change --g2 to fix.
        ax.scatter(df.loc[not_found, "beta"],
                   np.zeros(not_found.sum()),
                   marker="x", color="gray", s=80,
                   label=("no T_c in window: bump --T_hi "
                          f"(currently {args.T_hi}) or change --g2"))
    ax.axvline(args.beta_c, color="gray", linestyle=":",
               label=fr"$\beta_c \approx {args.beta_c:.3f}$")
    ax.set_xlabel(r"$\beta$"); ax.set_ylabel(r"$T_c$ (lattice units)")
    ax.set_title(rf"BCS $T_c$ from gauge-boson propagator   "
                 f"$g^2={args.g2}$, $\\mu={args.mu}$")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    plt.close(fig)
    print(f"Wrote plot   -> {args.out}")


if __name__ == "__main__":
    main()
