#!/usr/bin/env python3
"""Parallel beta-sweep driver for the SU(2) lattice gauge simulator.

Spawns multiple `su2lgt` processes concurrently across CPU cores, one per
(L_s, beta) point.  Each process is independent, writes its own CSV, and
gets a unique RNG seed derived from the base seed.

Replaces scripts/sweep.sh with a cross-platform tool that works the same
on Windows, macOS, and Linux.

Usage examples:
    python scripts/sweep.py                                       # defaults
    python scripts/sweep.py --workers 4                           # use 4 cores
    python scripts/sweep.py --sizes "6 8" --betas "2.2 2.3 2.4"   # quick test
    python scripts/sweep.py --prop_every 5                        # also extract G(r)

Environment:
    --binary    auto-detected if omitted; tries CMake/VS standard locations.
"""

import argparse
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


# Standard build-output locations the script will probe for su2lgt.
# Order matters: on each platform, the first matching candidate wins, so
# we list the .exe variants first (Windows) and the bare-name variants
# afterwards (Linux/macOS/Git Bash compiling natively).  At runtime we
# ALSO platform-filter, so a stray ./build/su2lgt left over from a
# Linux-style build attempt won't be picked up on Windows.
CANDIDATE_PATHS = [
    # Windows-style .exe outputs --------------------------------
    "./build/x64-Release/Release/su2lgt.exe",        # VS preset, MSBuild gen
    "./build/x64-Release/su2lgt.exe",                # VS preset, Ninja gen
    "./build/x64-Debug/Debug/su2lgt.exe",
    "./build/x64-Debug/su2lgt.exe",
    "./out/build/x64-Release/su2lgt.exe",            # VS default
    "./out/build/x64-RelWithDebInfo/su2lgt.exe",
    "./out/build/x64-Debug/su2lgt.exe",
    "./build/Release/su2lgt.exe",
    "./build/Debug/su2lgt.exe",
    "./build/su2lgt.exe",
    # Linux / macOS style bare-name outputs ---------------------
    "./build/su2lgt",
    "./build/linux-release/su2lgt",
    "./build/Release/su2lgt",
    "./build-release/su2lgt",
]


def find_binary(explicit=None):
    if explicit:
        p = Path(explicit).resolve()
        if p.is_file():
            return p
        sys.exit(f"Binary not found: {p}")

    is_windows = sys.platform == "win32"

    tried = []
    for c in CANDIDATE_PATHS:
        # Platform filter: only .exe on Windows, only non-.exe elsewhere
        if is_windows and not c.endswith(".exe"):
            continue
        if not is_windows and c.endswith(".exe"):
            continue
        p = Path(c)
        tried.append(c)
        if p.is_file():
            return p.resolve()

    # Last-ditch: glob the obvious roots for the right name
    target = "su2lgt.exe" if is_windows else "su2lgt"
    for root in ("build", "out/build"):
        for found in Path(root).rglob(target) if Path(root).is_dir() else []:
            if found.is_file():
                return found.resolve()
            tried.append(str(found))

    sys.exit(
        f"Could not locate the simulator binary (looking for '{target}').\n"
        f"Searched ({sys.platform}):\n"
        + "\n".join(f"    {c}" for c in tried)
        + "\n\nBuild first, or pass --binary PATH explicitly."
    )


SUMMARY_RE = {
    "absL":  r"^# <\|L\|>\s*=\s*([+-]?[\d.eE+-]+)",
    "U4":    r"^# Binder U4\s*=\s*([+-]?[\d.eE+-]+)",
    "chi":   r"^# susceptibility\s*=\s*([+-]?[\d.eE+-]+)",
    "plaq":  r"^# <plaq>\s*=\s*([+-]?[\d.eE+-]+)",
    "acc":   r"^# acceptance rate\s*=\s*([+-]?[\d.eE+-]+)",
}


def parse_summary(log_path):
    summary = {}
    try:
        text = Path(log_path).read_text()
    except OSError:
        return summary
    for k, pat in SUMMARY_RE.items():
        m = re.search(pat, text, re.MULTILINE)
        if m:
            summary[k] = float(m.group(1))
    return summary


def run_one(binary_str, sim_args, log_path_str):
    """Worker: launch the simulator with its log redirected to disk.

    Returns a dict with returncode, wall time, parsed summary."""
    t0 = time.time()
    with open(log_path_str, "w") as fh:
        proc = subprocess.run([binary_str, *sim_args],
                              stdout=fh, stderr=subprocess.STDOUT,
                              text=True)
    elapsed = time.time() - t0
    return {
        "returncode": proc.returncode,
        "elapsed":    elapsed,
        "summary":    parse_summary(log_path_str),
    }


def make_args(Ls, beta, args, seed):
    a = [
        "--Ls",       str(Ls),
        "--Lt",       str(args.Lt),
        "--beta",     f"{beta}",
        "--n_therm",  str(args.n_therm),
        "--n_meas",   str(args.n_meas),
        "--prop_eps", str(args.eps),
        "--n_hits",   str(args.n_hits),
        "--start",    args.start,
        "--seed",     hex(seed),
        "--out_dir",  str(args.out_dir),
    ]
    if args.prop_every > 0:
        a += ["--prop_every", str(args.prop_every)]
        a += ["--prop_max_iter", str(args.prop_max_iter)]
    if args.no_wilson:
        a += ["--no_wilson"]
    if args.label:
        a += ["--label", args.label]
    return a


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--binary",   default=None,
                   help="Simulator binary path (auto-detect if omitted)")
    p.add_argument("--out_dir",  default=Path("data"), type=Path)
    p.add_argument("--sizes",    default="8 12 16",
                   help='Spatial extents, space-separated (default "8 12 16")')
    p.add_argument("--betas",    default="2.10 2.18 2.22 2.24 2.26 2.28 "
                                          "2.30 2.32 2.34 2.36 2.40 2.50",
                   help="Beta values, space-separated")
    p.add_argument("--Lt",       type=int, default=4)
    p.add_argument("--n_therm",  type=int, default=2000)
    p.add_argument("--n_meas",   type=int, default=20000)
    p.add_argument("--n_hits",   type=int, default=5)
    p.add_argument("--eps",      type=float, default=0.45)
    p.add_argument("--start",    default="cold", choices=["cold", "hot"])
    p.add_argument("--prop_every",   type=int, default=0,
                   help="Measure boson propagator every K-th meas (0=off)")
    p.add_argument("--prop_max_iter", type=int, default=2000)
    p.add_argument("--no_wilson", action="store_true",
                   help="Skip Wilson / Polyakov-pair measurements")
    p.add_argument("--label",    default="",
                   help="Optional run label appended to output filenames")
    p.add_argument("--workers",  type=int, default=None,
                   help="Concurrent workers (default = cpu_count - 1)")
    p.add_argument("--base_seed", type=lambda s: int(s, 0),
                   default=0xC0FFEEBABEF00D,
                   help="Base RNG seed; per-job seeds are derived from it")
    args = p.parse_args()

    binary = find_binary(args.binary)
    workers = args.workers if args.workers else max(1, (os.cpu_count() or 2) - 1)
    sizes = [int(s) for s in args.sizes.split()]
    betas = [float(b) for b in args.betas.split()]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "logs").mkdir(parents=True, exist_ok=True)

    # Build the job list with deterministic per-job seeds.
    jobs = []
    for i, Ls in enumerate(sizes):
        for j, beta in enumerate(betas):
            seed = (args.base_seed + i * 100003 + j * 1009) & 0xFFFFFFFFFFFFFFFF
            sim_args = make_args(Ls, beta, args, seed)
            log_path = args.out_dir / "logs" / \
                f"Ls{Ls}_Lt{args.Lt}_beta{beta}.log"
            jobs.append((Ls, beta, sim_args, log_path))

    print(f"Sweep: {len(jobs)} jobs ({len(sizes)} sizes x "
          f"{len(betas)} betas), {workers} workers")
    print(f"  Binary:  {binary}")
    print(f"  Output:  {args.out_dir}")
    print(f"  Sweeps:  n_therm={args.n_therm}, n_meas={args.n_meas}, "
          f"meas/sweep config")
    if args.prop_every > 0:
        print(f"  Propagator: every {args.prop_every} measurements")
    print()

    t0 = time.time()
    rcs = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(run_one, str(binary), sim_args, str(log_path)):
                (Ls, beta)
            for (Ls, beta, sim_args, log_path) in jobs
        }
        n_done = 0
        for fut in as_completed(futures):
            Ls, beta = futures[fut]
            n_done += 1
            try:
                res = fut.result()
            except Exception as e:
                print(f"[{n_done:3d}/{len(jobs)}] EXCEPTION  "
                      f"Ls={Ls} beta={beta:.3f}: {e}")
                rcs.append(-1)
                continue

            rc      = res["returncode"]
            elapsed = res["elapsed"]
            s       = res["summary"]
            tag     = "ok  " if rc == 0 else "FAIL"
            absL    = s.get("absL", float("nan"))
            U4      = s.get("U4",   float("nan"))
            chi     = s.get("chi",  float("nan"))
            print(f"[{n_done:3d}/{len(jobs)}] {tag}  "
                  f"Ls={Ls:2d} beta={beta:.3f}  {elapsed:6.1f}s   "
                  f"<|L|>={absL:.4f}  U4={U4:+.4f}  chi={chi:.3f}")
            rcs.append(rc)

    total = time.time() - t0
    n_fail = sum(1 for rc in rcs if rc != 0)
    print()
    print(f"Sweep complete: {len(jobs) - n_fail} ok, {n_fail} failed in "
          f"{total:.1f}s wall, {workers} workers used")
    if n_fail == 0:
        print(f"Now run: python scripts/analyze.py {args.out_dir}")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
