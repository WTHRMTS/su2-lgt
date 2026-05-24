#!/usr/bin/env python3
"""Plot the gauge-boson propagator G(r) extracted from a CSV produced
by `su2lgt --prop_every K`.

Fits log G(r) vs r in the linear regime r in [1, Lx/4] to a Yukawa-like
form  G(r) ~ Z exp(-m r), reporting the screening mass m and amplitude Z.

This screening mass is the inverse correlation length of the spatial
gauge field in Landau gauge.  Across the deconfinement transition we
expect m to drop, signalling longer-range gauge propagation in the
deconfined phase.

Usage:
    python scripts/propagator.py path/to/run.csv [--out PATH]

Produces a two-panel plot:
    - log G(r) vs r with linear fit overlaid
    - G(r) on linear axes (so structure near r=0 is visible)
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


def parse_header(path):
    cfg = {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            for m in re.finditer(r"(\w+)=([\w.+\-]+)", line):
                cfg[m.group(1)] = m.group(2)
    return cfg


def collect_propagator(df, r_max):
    means, errs = [], []
    for r in range(r_max + 1):
        col = f"Gprop_r{r}"
        if col not in df.columns:
            return None, None
        vals = df[col].dropna().values
        if len(vals) == 0:
            return None, None
        means.append(vals.mean())
        errs.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1
                    else 0.0)
    return np.asarray(means), np.asarray(errs)


def fit_screening_mass(rs, Gs, r_lo=1, r_hi=None):
    """Fit log G(r) = -m r + log Z over a chosen r range."""
    if r_hi is None:
        r_hi = len(Gs) - 1
    mask = (rs >= r_lo) & (rs <= r_hi) & (Gs > 0)
    if mask.sum() < 2:
        return None, None, None, None
    x = rs[mask].astype(float)
    y = np.log(Gs[mask])
    coef, cov = np.polyfit(x, y, 1, cov=True)
    slope, intercept = coef
    m  = -slope
    Z  = np.exp(intercept)
    sm = float(np.sqrt(cov[0, 0]))
    return m, sm, Z, mask


def make_plots(cfg, df, out_path):
    r_max = int(cfg.get("r_max", 0))
    Lx    = int(cfg.get("Lx", 0))
    Lt    = int(cfg.get("Lt", 0))
    beta  = float(cfg.get("beta", 0))

    if cfg.get("propagator", "0") != "1":
        sys.exit("CSV does not contain propagator measurements "
                 "(was the run done with --prop_every 0?)")

    G_mean, G_err = collect_propagator(df, r_max)
    if G_mean is None:
        sys.exit("Could not load Gprop_r* columns from CSV.")

    rs = np.arange(r_max + 1)

    # Fit on the "physical" range that excludes the periodic-image flatten
    # at r ~ Lx/2.  Empirically [1, Lx/4 + 1] is a clean middle band.
    r_hi_fit = max(2, Lx // 4 + 1)
    m, sm, Z, mask = fit_screening_mass(rs, G_mean, r_lo=1, r_hi=r_hi_fit)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Panel 1: log G(r) with fit ----
    ax = axes[0]
    G_safe = np.clip(G_mean, 1e-30, None)
    ax.errorbar(rs, G_safe, yerr=G_err, fmt="o", capsize=3,
                label="G(r) data")
    if m is not None and mask is not None:
        rfit = rs[mask].astype(float)
        ax.plot(rfit, Z * np.exp(-m * rfit), "--",
                color="tab:red",
                label=fr"fit: $m={m:.3f} \pm {sm:.3f}$, $Z={Z:.3f}$")
    ax.set_yscale("log")
    ax.set_xlabel("r (lattice units)")
    ax.set_ylabel(r"$G(r)$")
    ax.set_title(rf"Gauge-boson static propagator   "
                 f"($L_s={Lx}$, $L_t={Lt}$, $\\beta={beta:.3f}$)")
    ax.legend()
    ax.grid(which="both", alpha=0.3)

    # ---- Panel 2: linear-axis G(r) ----
    ax = axes[1]
    ax.errorbar(rs, G_mean, yerr=G_err, fmt="o-", capsize=3)
    ax.set_xlabel("r")
    ax.set_ylabel(r"$G(r)$")
    ax.set_title("G(r) on linear axes")
    ax.grid(alpha=0.3)
    if m is not None:
        m_pretty = f"{m:.3f} ± {sm:.3f}"
        ax.text(0.6, 0.9, f"screening mass\n$m_a = {m_pretty}$\n"
                          fr"correlation length $\xi = 1/m = {1/m:.2f}$ a",
                transform=ax.transAxes,
                bbox=dict(facecolor="white", alpha=0.8),
                fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path, m, sm, Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="Output PNG path (default: alongside CSV)")
    args = ap.parse_args()
    if not args.csv.exists():
        sys.exit(f"No such CSV: {args.csv}")

    cfg = parse_header(args.csv)
    df = pd.read_csv(args.csv, comment="#")
    out = args.out or args.csv.with_suffix(".propagator.png")
    out_path, m, sm, Z = make_plots(cfg, df, out)

    print(f"Wrote {out_path}")
    if m is not None:
        beta = float(cfg.get("beta", 0))
        print(f"\n  beta = {beta:.4f}")
        print(f"  screening mass m_a = {m:.4f} +/- {sm:.4f} (lattice units)")
        print(f"  correlation length xi = {1/m:.3f} a")
        print(f"  amplitude        Z   = {Z:.4f}")


if __name__ == "__main__":
    main()
