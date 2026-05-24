#!/usr/bin/env python3
"""End-to-end physics verification of the SU(2) lattice gauge simulator.

Runs the C++ binary at well-known validation points and compares the
measured observables to either:

  * Analytic series expansions of the mean plaquette (strong-coupling
    SU(2) character expansion, asymptotic-freedom weak-coupling series).
  * Qualitative limits of the order parameter (small <|L|> deep in the
    confined phase, U_4 -> 2/3 deep in the deconfined phase, etc.).
  * Published values of the mean plaquette at beta_c on the same
    Ns^3 x Nt geometry.

Each check prints PASS or FAIL with the measured value and target band.
The script exits non-zero if any check fails.

Usage:
    python scripts/verify.py [--binary PATH]

Defaults to ./build/su2lgt; build first with cmake or the provided
g++ command.  The whole suite takes ~30 s on modest hardware.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


# ---- subprocess plumbing ------------------------------------------------

def run_simulator(binary, args):
    cmd = [str(binary), '--no_csv', *args]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"\nSimulator failed:\n{e.output}\n")
        raise
    return parse_summary(out), out


SUMMARY_PATTERNS = [
    ('beta',   r'^# beta\s*=\s*([+-]?[\d.eE+-]+)'),
    ('acc',    r'^# acceptance rate\s*=\s*([+-]?[\d.eE+-]+)'),
    ('plaq',   r'^# <plaq>\s*=\s*([+-]?[\d.eE+-]+)'),
    ('L',      r'^# <L>\s*=\s*([+-]?[\d.eE+-]+)'),
    ('absL',   r'^# <\|L\|>\s*=\s*([+-]?[\d.eE+-]+)'),
    ('L2',     r'^# <L\^2>\s*=\s*([+-]?[\d.eE+-]+)'),
    ('L4',     r'^# <L\^4>\s*=\s*([+-]?[\d.eE+-]+)'),
    ('binder', r'^# Binder U4\s*=\s*([+-]?[\d.eE+-]+)'),
    ('chi',    r'^# susceptibility\s*=\s*([+-]?[\d.eE+-]+)'),
]


def parse_summary(text):
    out = {}
    for key, patt in SUMMARY_PATTERNS:
        m = re.search(patt, text, re.MULTILINE)
        if m:
            out[key] = float(m.group(1))
    return out


# ---- pass / fail bookkeeping --------------------------------------------

n_pass = 0
n_fail = 0


def check(label, ok, detail=''):
    global n_pass, n_fail
    if ok:
        n_pass += 1
        print(f'  PASS  {label}    {detail}')
    else:
        n_fail += 1
        print(f'  FAIL  {label}    {detail}')


# ---- physics cases ------------------------------------------------------

def case_strong_coupling(binary):
    """Strong-coupling expansion: SU(2) character expansion gives
        <P> = beta/4 + beta^3/96 + O(beta^5)
    in the small-beta regime.  Verify at beta=0.5 and beta=1.0 on a
    4^4 box where statistics are quick."""
    print('\n[A] Strong-coupling expansion')
    for beta in (0.5, 1.0):
        s, _ = run_simulator(binary, [
            '--Ls', '4', '--Lt', '4',
            '--beta', f'{beta}',
            '--n_therm', '300', '--n_meas', '1500',
            '--prop_eps', '0.6', '--start', 'cold',
            '--seed', '0xA1B2C3'])
        target = beta / 4.0 + (beta ** 3) / 96.0
        delta = s['plaq'] - target
        ok = abs(delta) < 0.04
        check(f'beta={beta:.1f}: <P>={s["plaq"]:.4f}  series={target:.4f}',
              ok, f'(delta={delta:+.4f})')


def case_weak_coupling(binary):
    """Weak-coupling expansion (asymptotic freedom): for SU(2),
        <P> = 1 - 3/(4 beta) - 0.0469/beta^2 + O(1/beta^3)
    deep in the weak-coupling regime."""
    print('\n[B] Weak-coupling expansion')
    for beta, eps in [(4.0, 0.30), (8.0, 0.20)]:
        s, _ = run_simulator(binary, [
            '--Ls', '4', '--Lt', '4',
            '--beta', f'{beta}',
            '--n_therm', '500', '--n_meas', '2000',
            '--prop_eps', f'{eps}', '--start', 'cold',
            '--seed', '0xD4E5F6'])
        target = 1.0 - 3.0 / (4.0 * beta) - 0.0469 / (beta ** 2)
        delta = s['plaq'] - target
        ok = abs(delta) < 0.02
        check(f'beta={beta:.1f}: <P>={s["plaq"]:.4f}  series={target:.4f}',
              ok, f'(delta={delta:+.4f})')


def case_polyakov_limits(binary):
    """Polyakov-loop limits on Ns^3 x 4: deep confined phase (beta=1.8)
    has small <|L|> and Binder ~ 0; deep deconfined phase (beta=2.7)
    has large <|L|> and Binder approaching 2/3."""
    print('\n[C] Polyakov-loop phase limits')

    s_lo, _ = run_simulator(binary, [
        '--Ls', '6', '--Lt', '4', '--beta', '1.8',
        '--n_therm', '400', '--n_meas', '1500',
        '--prop_eps', '0.50', '--start', 'cold',
        '--seed', '0xC0FFEE01'])
    check('confined: <|L|> < 0.10 at beta=1.8',
          s_lo['absL'] < 0.10,
          f'(<|L|>={s_lo["absL"]:.4f})')
    check('confined: |U_4| < 0.30 at beta=1.8',
          abs(s_lo['binder']) < 0.30,
          f'(U_4={s_lo["binder"]:+.4f})')

    s_hi, _ = run_simulator(binary, [
        '--Ls', '6', '--Lt', '4', '--beta', '2.7',
        '--n_therm', '400', '--n_meas', '1500',
        '--prop_eps', '0.40', '--start', 'cold',
        '--seed', '0xC0FFEE02'])
    check('deconfined: <|L|> > 0.30 at beta=2.7',
          s_hi['absL'] > 0.30,
          f'(<|L|>={s_hi["absL"]:.4f})')
    check('deconfined: U_4 > 0.55 at beta=2.7',
          s_hi['binder'] > 0.55,
          f'(U_4={s_hi["binder"]:+.4f})')


def case_critical_plaquette(binary):
    """At the published Nt=4 critical coupling beta_c ~= 2.298, the
    mean plaquette on 6^3 x 4 lies in roughly 0.58 - 0.62."""
    print('\n[D] Mean plaquette near beta_c on Nt=4')
    s, _ = run_simulator(binary, [
        '--Ls', '6', '--Lt', '4', '--beta', '2.30',
        '--n_therm', '500', '--n_meas', '2500',
        '--prop_eps', '0.45', '--start', 'cold',
        '--seed', '0x12345678'])
    check('6^3 x 4, beta=2.30: <P> in [0.58, 0.62]',
          0.58 <= s['plaq'] <= 0.62,
          f'(<P>={s["plaq"]:.4f})')


def case_acceptance_rate(binary):
    """Sanity: with the recommended proposal width the acceptance
    rate should land in (0.30, 0.85) on a hot-equilibrium chain."""
    print('\n[E] Acceptance-rate sanity')
    s, _ = run_simulator(binary, [
        '--Ls', '4', '--Lt', '4', '--beta', '2.30',
        '--n_therm', '200', '--n_meas', '500',
        '--prop_eps', '0.45', '--start', 'cold',
        '--seed', '0xACE'])
    ok = 0.30 < s['acc'] < 0.85
    check(f'acceptance in (0.30, 0.85) at eps=0.45',
          ok, f'(acc={s["acc"]:.3f})')


# ---- driver -------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--binary', default='./build/su2lgt')
    args = p.parse_args()

    binary = Path(args.binary)
    if not binary.is_file():
        sys.exit(f'Binary not found: {binary}.  Build first.')

    print(f'Verifying {binary}')

    case_strong_coupling(binary)
    case_weak_coupling(binary)
    case_polyakov_limits(binary)
    case_critical_plaquette(binary)
    case_acceptance_rate(binary)

    print()
    print(f'=== {n_pass} passed, {n_fail} failed ===')
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == '__main__':
    main()
