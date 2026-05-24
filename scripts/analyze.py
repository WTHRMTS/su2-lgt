#!/usr/bin/env python3
"""Analyze SU(2) lattice gauge theory beta-sweep CSVs.

Reads every *.csv in the given data directory, applies binning + jackknife
to each run to get correctly-error-barred Polyakov-loop moments, Binder
cumulant U_4 and susceptibility chi_L, and produces a four-panel diagnostic
plot:

    1. <|L|>(beta) for each Ls
    2. Binder U_4(beta) for each Ls            <- curves cross at beta_c
    3. chi_L(beta) for each Ls                 <- peaks locate beta_c
    4. log-log scaling of peak chi vs Ls       <- slope ~ gamma/nu

Usage:
    python scripts/analyze.py [data_dir]      (default: ./data)

Outputs:
    <data_dir>/aggregated.csv
    <data_dir>/phase_diagnostic.png
"""

import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# CSV loading
# --------------------------------------------------------------------------

def parse_header(path):
    """Parse the first comment line containing key=value pairs."""
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


# --------------------------------------------------------------------------
# Binning + jackknife
# --------------------------------------------------------------------------

def block(values, n_blocks):
    """Block-average `values` into `n_blocks` bins."""
    values = np.asarray(values)
    N = len(values)
    if N == 0 or n_blocks <= 0:
        return values.copy()
    bs = max(1, N // n_blocks)
    n_blocks = N // bs
    return values[: n_blocks * bs].reshape(n_blocks, bs).mean(axis=1)


def stderr_of_mean(blocks):
    if len(blocks) < 2:
        return 0.0
    return float(np.std(blocks, ddof=1) / np.sqrt(len(blocks)))


def jackknife(blocks_dict, fn):
    """Leave-one-out jackknife.  `blocks_dict` is {name: 1D array of blocks},
    and `fn` is a function that maps that dict to a scalar.

    Returns (full_value, jackknife_error)."""
    keys = list(blocks_dict.keys())
    nb = len(blocks_dict[keys[0]])
    full = fn(blocks_dict)
    leaves = np.empty(nb)
    for i in range(nb):
        sub = {k: np.delete(blocks_dict[k], i) for k in keys}
        leaves[i] = fn(sub)
    err = float(np.sqrt((nb - 1) / nb * np.sum((leaves - leaves.mean()) ** 2)))
    return float(full), err


def run_observables(df, Vs, n_blocks=20):
    blocks = {
        "L":    block(df["L"].values,    n_blocks),
        "absL": block(df["absL"].values, n_blocks),
        "L2":   block(df["L2"].values,   n_blocks),
        "L4":   block(df["L4"].values,   n_blocks),
        "plaq": block(df["plaq"].values, n_blocks),
    }
    nb = len(blocks["L"])

    def U4(b):
        m2 = b["L2"].mean()
        m4 = b["L4"].mean()
        return 1.0 - m4 / (3.0 * m2 * m2) if m2 > 0 else 0.0

    def chi(b):
        return Vs * (b["L2"].mean() - b["absL"].mean() ** 2)

    U4_full, err_U4   = jackknife(blocks, U4)
    chi_full, err_chi = jackknife(blocks, chi)

    return {
        "n_records": len(df),
        "n_blocks":  nb,
        "mean_L":    float(blocks["L"].mean()),    "err_L":    stderr_of_mean(blocks["L"]),
        "mean_absL": float(blocks["absL"].mean()), "err_absL": stderr_of_mean(blocks["absL"]),
        "mean_L2":   float(blocks["L2"].mean()),   "err_L2":   stderr_of_mean(blocks["L2"]),
        "mean_L4":   float(blocks["L4"].mean()),   "err_L4":   stderr_of_mean(blocks["L4"]),
        "mean_plaq": float(blocks["plaq"].mean()), "err_plaq": stderr_of_mean(blocks["plaq"]),
        "U4":        U4_full,                      "err_U4":   err_U4,
        "chi":       chi_full,                     "err_chi":  err_chi,
    }


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def aggregate(data_dir):
    rows = []
    # Skip files this script (or its siblings) emit, not files it ingests.
    NON_RUN_NAMES = {"aggregated.csv", "aggregated_smoke.csv",
                     "Tc_vs_beta.csv"}
    NON_RUN_PREFIXES = ("aggregated", "Tc_vs_beta", "summary",
                        "gap_diagnostic", "gap_g")
    for csv in sorted(Path(data_dir).glob("*.csv")):
        if csv.name in NON_RUN_NAMES:
            continue
        if any(csv.name.startswith(prefix) for prefix in NON_RUN_PREFIXES):
            continue
        cfg, df = load_run(csv)
        if df.empty:
            print(f"  (skip empty: {csv.name})")
            continue
        try:
            Lx = int(cfg["Lx"]); Ly = int(cfg["Ly"])
            Lz = int(cfg["Lz"]); Lt = int(cfg["Lt"])
            beta = float(cfg["beta"])
        except KeyError as e:
            print(f"  (skip {csv.name}: missing header field {e})")
            continue
        Vs = Lx * Ly * Lz
        obs = run_observables(df, Vs)
        rows.append({"Ls": Lx, "Lt": Lt, "beta": beta, "Vs": Vs,
                     "csv": csv.name, **obs})
    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows)
              .sort_values(["Ls", "beta"])
              .reset_index(drop=True))


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def make_plots(res, out_path):
    sizes = sorted(res["Ls"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1. <|L|>(beta) -- the visible order parameter
    ax = axes[0, 0]
    for Ls in sizes:
        s = res[res["Ls"] == Ls].sort_values("beta")
        ax.errorbar(s["beta"], s["mean_absL"], yerr=s["err_absL"],
                    fmt="o-", capsize=3, label=f"$L_s={Ls}$")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\langle |L| \rangle$")
    ax.set_title("Polyakov-loop order parameter")
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Binder cumulant -- curves cross at beta_c
    ax = axes[0, 1]
    for Ls in sizes:
        s = res[res["Ls"] == Ls].sort_values("beta")
        ax.errorbar(s["beta"], s["U4"], yerr=s["err_U4"],
                    fmt="o-", capsize=3, label=f"$L_s={Ls}$")
    ax.axhline(0.0,    color="gray", linestyle=":", alpha=0.7)
    ax.axhline(2/3,    color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$U_4 = 1 - \langle L^4 \rangle / (3\,\langle L^2 \rangle^2)$")
    ax.set_title(r"Binder cumulant — curves cross at $\beta_c$")
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. susceptibility chi_L
    ax = axes[1, 0]
    for Ls in sizes:
        s = res[res["Ls"] == Ls].sort_values("beta")
        ax.errorbar(s["beta"], s["chi"], yerr=s["err_chi"],
                    fmt="o-", capsize=3, label=f"$L_s={Ls}$")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\chi_L = V_s\,(\langle L^2 \rangle - \langle |L| \rangle^2)$")
    ax.set_title("Polyakov-loop susceptibility")
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. peak chi vs Ls (log-log)  --  slope ~ gamma/nu of universality class
    ax = axes[1, 1]
    peaks = []
    for Ls in sizes:
        s = res[res["Ls"] == Ls]
        if len(s) == 0:
            continue
        i = s["chi"].idxmax()
        peaks.append((Ls, float(s.loc[i, "chi"]), float(s.loc[i, "err_chi"]),
                      float(s.loc[i, "beta"])))
    if len(peaks) >= 2:
        Ls_vals = np.array([p[0] for p in peaks])
        chi_pk  = np.array([p[1] for p in peaks])
        chi_err = np.array([p[2] for p in peaks])
        ax.errorbar(Ls_vals, chi_pk, yerr=chi_err, fmt="o", capsize=4)
        log_Ls = np.log(Ls_vals)
        log_chi = np.log(chi_pk)
        slope, intercept = np.polyfit(log_Ls, log_chi, 1)
        Ls_fit = np.array([Ls_vals.min(), Ls_vals.max()])
        ax.plot(Ls_fit, np.exp(slope * np.log(Ls_fit) + intercept),
                "--", alpha=0.7, label=f"fit slope = {slope:.2f}")
        ax.set_title(f"$\\chi_L^{{\\rm max}}$ scaling   "
                     f"(3D Ising: $\\gamma/\\nu \\approx 1.97$)")
        ax.legend()
    ax.set_xlabel(r"$L_s$")
    ax.set_ylabel(r"$\chi_L^{\rm max}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main():
    data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    if not data_dir.exists():
        sys.exit(f"No such directory: {data_dir}")

    res = aggregate(data_dir)
    if res.empty:
        sys.exit(f"No usable CSVs found in {data_dir}/")

    cols = ["Ls", "beta", "n_records", "mean_absL", "err_absL",
            "U4", "err_U4", "chi", "err_chi"]
    print(res[cols].to_string(index=False))

    agg_path = data_dir / "aggregated.csv"
    res.to_csv(agg_path, index=False)
    print(f"\nWrote {agg_path}")

    plot_path = make_plots(res, data_dir / "phase_diagnostic.png")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
