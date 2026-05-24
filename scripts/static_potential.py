#!/usr/bin/env python3
"""Extract the static-quark potential V(r) from a single CSV produced by
the SU(2) lattice gauge simulator (Step 10).

Two independent estimates are produced from each run:

    1.  V_P(r)  from the Polyakov-loop pair correlator at finite T:
            P(r) = <l(0) l(r)>     ~     exp(-V(r) / T_phys)
        At fixed N_t, V(r)/T_phys = -log(P(r) - <L>^2)  (offset by the
        Z_2 vacuum value in the deconfined phase).

    2.  V_W(R)  from the Wilson loop area-law fit:
            W(R, T)  ~  exp(-V(R) T) * (excited contributions)
        For each R, fit log(W(R, T)) vs T linearly; slope = -V(R).

A linear fit of V_W(R) vs R gives the spatial string tension sigma.

Usage:
    python scripts/static_potential.py path/to/run.csv [--out PATH]

Produces a four-panel plot:
    - log P(r) vs r  (with <L>^2 asymptote shown)
    - V_P(r)/T  vs r  (extracted potential at finite T)
    - log W(R, T) vs T for each R  (Wilson lines)
    - V_W(R) vs R  with linear fit (string tension sigma)
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


def load_run(path):
    cfg = parse_header(path)
    df = pd.read_csv(path, comment="#")
    return cfg, df


def collect_pcorr(df, r_max):
    """Return mean P(r) and standard error per r."""
    means, errs = [], []
    for r in range(r_max + 1):
        col = f"Pcorr_r{r}"
        if col not in df.columns:
            return None, None
        vals = df[col].values
        means.append(vals.mean())
        errs.append(vals.std(ddof=1) / np.sqrt(len(vals)))
    return np.asarray(means), np.asarray(errs)


def collect_wilson(df, R_max, T_max):
    """Return W(R, T) means with R in 1..R_max, T in 1..T_max."""
    W = np.zeros((R_max + 1, T_max + 1))
    for R in range(1, R_max + 1):
        for T in range(1, T_max + 1):
            col = f"W_R{R}_T{T}"
            if col not in df.columns:
                return None
            W[R, T] = df[col].values.mean()
    return W


def fit_V_from_W(W, R_max, T_max, T_min=1):
    """For each R, fit log W(R, T) = -V(R) * T - log Z(R) by least squares
    over T in [T_min, T_max], and return V(R) and standard error."""
    V, Verr = np.zeros(R_max + 1), np.zeros(R_max + 1)
    Ts = np.arange(T_min, T_max + 1, dtype=float)
    for R in range(1, R_max + 1):
        Ws = W[R, T_min:T_max + 1]
        if np.any(Ws <= 0):
            V[R] = np.nan; Verr[R] = np.nan
            continue
        y = np.log(Ws)
        # Linear regression: y = a*T + b ; V = -a
        A = np.vstack([Ts, np.ones_like(Ts)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        V[R] = -coef[0]
        # naive 1-sigma from residuals
        resid = y - A @ coef
        if len(Ts) > 2:
            sigma = np.sqrt(np.sum(resid ** 2) / (len(Ts) - 2))
            S_T = np.sum((Ts - Ts.mean()) ** 2)
            Verr[R] = sigma / np.sqrt(S_T)
        else:
            Verr[R] = 0.0
    return V, Verr


def make_plots(cfg, df, out_path):
    r_max = int(cfg.get("r_max", 0))
    R_max = int(cfg.get("R_max", 0))
    T_max = int(cfg.get("T_max", 0))
    Lt    = int(cfg.get("Lt", 4))
    beta  = float(cfg.get("beta", 0))
    Lx    = int(cfg.get("Lx", 0))

    P_mean, P_err = collect_pcorr(df, r_max)
    W = collect_wilson(df, R_max, T_max)

    if P_mean is None or W is None:
        sys.exit("CSV is missing Wilson / pair-correlator columns "
                 "(was the run done with --no_wilson?)")

    # Mean L from CSV (consistent with how main.cpp summarises)
    L_mean = df["L"].mean()
    L_asym = L_mean * L_mean   # <L>^2

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # ---- Panel 1: log P(r) vs r ----
    ax = axes[0, 0]
    rs = np.arange(r_max + 1)
    P_safe = np.clip(P_mean, 1e-15, None)
    ax.errorbar(rs, P_safe, yerr=P_err, fmt="o-",
                capsize=3, label=r"$P(r) = \langle\ell(0)\ell(r)\rangle$")
    ax.axhline(L_asym, color="gray", linestyle=":",
               label=fr"$\langle L\rangle^2 = {L_asym:.4f}$")
    ax.set_xlabel("r (lattice units)")
    ax.set_ylabel(r"$P(r)$")
    ax.set_yscale("log")
    ax.set_title(f"Polyakov-loop pair correlator   (Ls={Lx}, Lt={Lt}, "
                 fr"$\beta$={beta:.3f})")
    ax.legend()
    ax.grid(which="both", alpha=0.3)

    # ---- Panel 2: V_P(r) / T  =  -log(P(r) - <L>^2)  with positivity ----
    ax = axes[0, 1]
    P_sub = P_mean - L_asym
    P_pos = np.where(P_sub > 0, P_sub, np.nan)
    V_over_T = -np.log(P_pos)
    ax.plot(rs, V_over_T, "o-")
    ax.set_xlabel("r")
    ax.set_ylabel(r"$V(r) / T_{\rm phys}$")
    ax.set_title(r"Static potential from Polyakov pair correlator")
    ax.grid(alpha=0.3)

    # ---- Panel 3: log W(R, T) vs T per R ----
    ax = axes[1, 0]
    Ts = np.arange(1, T_max + 1)
    for R in range(1, R_max + 1):
        Ws = W[R, 1:T_max + 1]
        ax.plot(Ts, np.log(Ws), "o-", label=f"R={R}")
    ax.set_xlabel("T")
    ax.set_ylabel(r"$\log W(R, T)$")
    ax.set_title("Wilson loop decay (slope per R = -V(R))")
    ax.legend(ncol=2, fontsize="small")
    ax.grid(alpha=0.3)

    # ---- Panel 4: V_W(R) with linear fit (string tension) ----
    ax = axes[1, 1]
    V, Verr = fit_V_from_W(W, R_max, T_max)
    Rs = np.arange(1, R_max + 1)
    ax.errorbar(Rs, V[1:], yerr=Verr[1:], fmt="o", capsize=4)
    # Linear fit V(R) ~ sigma R + V0   on R >= 2 (drop short-distance Coulomb)
    if R_max >= 3:
        mask = (Rs >= 2) & np.isfinite(V[1:])
        if mask.sum() >= 2:
            coef = np.polyfit(Rs[mask], V[1:][mask], 1)
            sigma = coef[0]
            ax.plot(Rs, np.polyval(coef, Rs), "--",
                    alpha=0.7, label=fr"slope $\sigma$ = {sigma:.3f}")
            ax.legend()
    ax.set_xlabel("R")
    ax.set_ylabel(r"$V(R)$ (lattice units)")
    ax.set_title("Static potential from Wilson loops")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="Output PNG path (default: alongside CSV)")
    args = ap.parse_args()
    if not args.csv.exists():
        sys.exit(f"No such CSV: {args.csv}")

    cfg, df = load_run(args.csv)
    out = args.out or args.csv.with_suffix(".potential.png")
    make_plots(cfg, df, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
