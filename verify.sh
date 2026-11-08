#!/usr/bin/env bash
# Run fast repository checks or the user-owned solver regression lane.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NP=4
LANE=fast
usage() { echo 'Usage: bash verify.sh [NP] [--lane fast|solver|all]'; }
while (( $# )); do
    case "$1" in
        --lane) (( $# >= 2 )) || { usage; exit 2; }; LANE=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *[!0-9]*|'') usage; exit 2 ;;
        *) NP=$1; shift ;;
    esac
done
(( NP > 0 )) || { usage; exit 2; }
case "$LANE" in fast|solver|all) ;; *) usage; exit 2 ;; esac

export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/mdao-su2-matplotlib}"

step() {
    local name=$1
    shift
    echo "--- $name ---"
    if "$@"; then echo "PASS $name"; else echo "FAIL $name" >&2; return 1; fi
}

cd "$REPO"
if [[ $LANE == fast || $LANE == all ]]; then
    step 'Environment' python "$REPO/tools/check_env.py"
    step 'Regression fast' python "$REPO/tests/regression.py" --lane fast
    step 'Shipped-config resolve' python -c '
from pathlib import Path
from mdao_su2.run_config import load_config, resolve_dict
paths = sorted(Path("cases").glob("*/input/configs/*.toml"))
if not paths:
    raise SystemExit("no runnable configurations")
for path in paths:
    resolve_dict(path)
    load_config(path, check_files=False)
print(f"resolved {len(paths)} runnable configurations")'
fi

if [[ $LANE == solver || $LANE == all ]]; then
    solver_out="${MDAO_SU2_REGRESSION_OUT:-${TMPDIR:-/tmp}/mdao-su2-regression-solver}"
    prefix=$(bash "$REPO/tools/bind_cores.sh" "$NP" | tail -1)
    read -r -a prefix_args <<< "$prefix"
    step 'Regression solver' "${prefix_args[@]}" python \
        "$REPO/tests/regression.py" --lane solver --np "$NP" --out "$solver_out"
fi

echo "PASSED (lane: $LANE, np=$NP)"
