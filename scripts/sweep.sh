#!/usr/bin/env bash
# Run a beta-sweep across multiple lattice sizes and dump CSVs.
#
# Usage:
#   ./scripts/sweep.sh                                # default sweep
#   N_THERM=500 N_MEAS=5000 ./scripts/sweep.sh        # quick check
#   SIZES="6 10" BETAS="2.20 2.30 2.40" ./scripts/sweep.sh
#
# Environment overrides:
#   SU2LGT     path to the binary (auto-detected if unset)
#   OUT_DIR    output directory for CSVs and logs (default data/)
#   N_THERM    thermalisation sweeps per run (default 2000)
#   N_MEAS     measurement sweeps per run (default 20000)
#   EPS        Metropolis proposal width (default 0.45)
#   SIZES      space-separated spatial extents (default "8 12 16")
#   BETAS      space-separated beta values
#   LT         temporal extent (default 4)

set -euo pipefail

OUT_DIR="${OUT_DIR:-data}"
N_THERM="${N_THERM:-2000}"
N_MEAS="${N_MEAS:-20000}"
EPS="${EPS:-0.45}"
LT="${LT:-4}"
SIZES="${SIZES:-8 12 16}"
BETAS="${BETAS:-2.10 2.18 2.22 2.24 2.26 2.28 2.30 2.32 2.34 2.36 2.40 2.50}"

# --- Locate the simulator binary ----------------------------------------
# If SU2LGT isn't set explicitly, scan a few common build locations:
#   - Linux / macOS / Git Bash CLI build:    ./build/su2lgt
#   - Same with .exe (Windows):              ./build/su2lgt.exe
#   - Visual Studio CMake (Release):         ./out/build/x64-Release/su2lgt.exe
#   - Visual Studio CMake (Debug):           ./out/build/x64-Debug/su2lgt.exe
#   - VS RelWithDebInfo:                     ./out/build/x64-RelWithDebInfo/su2lgt.exe

if [[ -z "${SU2LGT:-}" ]]; then
    for candidate in \
        ./build/su2lgt \
        ./build/su2lgt.exe \
        ./build/linux-release/su2lgt \
        ./build/x64-Release/Release/su2lgt.exe \
        ./build/x64-Release/su2lgt.exe \
        ./build/x64-Debug/Debug/su2lgt.exe \
        ./build/x64-Debug/su2lgt.exe \
        ./out/build/x64-Release/su2lgt.exe \
        ./out/build/x64-RelWithDebInfo/su2lgt.exe \
        ./out/build/x64-Debug/su2lgt.exe ; do
        if [[ -x "$candidate" ]]; then
            SU2LGT="$candidate"
            break
        fi
    done
fi

if [[ -z "${SU2LGT:-}" || ! -x "$SU2LGT" ]]; then
    echo "ERROR: simulator binary not found." >&2
    echo "Searched:" >&2
    echo "    ./build/su2lgt" >&2
    echo "    ./build/su2lgt.exe" >&2
    echo "    ./out/build/x64-Release/su2lgt.exe" >&2
    echo "    ./out/build/x64-RelWithDebInfo/su2lgt.exe" >&2
    echo "    ./out/build/x64-Debug/su2lgt.exe" >&2
    echo "" >&2
    echo "Either build first:" >&2
    echo "    cmake --build build -j                                # CLI" >&2
    echo "    Build All in Visual Studio                            # IDE" >&2
    echo "Or set SU2LGT explicitly:" >&2
    echo "    SU2LGT=path/to/su2lgt.exe ./scripts/sweep.sh" >&2
    exit 1
fi

mkdir -p "$OUT_DIR" "$OUT_DIR/logs"

echo "Using binary: $SU2LGT"
echo "Sweep:        SIZES=[$SIZES] BETAS=[$BETAS] Lt=$LT"
echo "              n_therm=$N_THERM n_meas=$N_MEAS eps=$EPS"
echo "              out_dir=$OUT_DIR"
echo

count=0
total=$(( $(echo $SIZES | wc -w) * $(echo $BETAS | wc -w) ))

for Ls in $SIZES; do
    for beta in $BETAS; do
        count=$((count + 1))
        log="$OUT_DIR/logs/Ls${Ls}_Lt${LT}_beta${beta}.log"
        printf '[%s] (%d/%d) Ls=%s beta=%s\n' \
               "$(date +%T)" "$count" "$total" "$Ls" "$beta"
        "$SU2LGT" --Ls "$Ls" --Lt "$LT" --beta "$beta" \
                  --n_therm "$N_THERM" --n_meas "$N_MEAS" \
                  --prop_eps "$EPS" --start cold \
                  --out_dir "$OUT_DIR" \
                  > "$log" 2>&1
        grep -E '^# (Binder U4|susceptibility|<\|L\|>)' "$log" \
            | sed 's/^/    /'
    done
done

echo
echo "Sweep complete.  CSVs in $OUT_DIR/"
echo "Run:  python3 scripts/analyze.py $OUT_DIR"
