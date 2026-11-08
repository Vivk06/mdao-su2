#!/usr/bin/env bash
# Pure-SU2 RAE 2822 optimization (SU2 tutorial condition, M=0.729, Re=6.5E6) cross-check. Self-contained.
set -euo pipefail
cd "$(dirname "$0")"

SU2_RUN="${SU2_RUN:-${SU2_PREFIX:-$HOME/Shape_Optimisation/SU2v8.5/Binaries}/bin}"
OPENMPI_LIB="${OPENMPI_LIB:-${OPENMPI_PREFIX:-$HOME/Shape_Optimisation/openmpi}/lib}"

export SU2_RUN
export PATH=$SU2_RUN:$PATH
export PYTHONPATH=$SU2_RUN:${PYTHONPATH:-}
if [ -n "$OPENMPI_LIB" ]; then
    export LD_LIBRARY_PATH=$OPENMPI_LIB:${LD_LIBRARY_PATH:-}
fi

if [ ! -f mesh_RAE2822_turb.su2 ]; then
    echo "ERROR: mesh_RAE2822_turb.su2 not found here -- run ./stage.sh first"
    echo "       (inside the repo checkout) or copy the mesh in manually."
    exit 1
fi

NP=${1:-12}
echo "SU2_RUN       = $SU2_RUN"
echo "OPENMPI_LIB   = ${OPENMPI_LIB:-<system MPI>}"
echo "np            = $NP"

command -v SU2_CFD >/dev/null 2>&1 || { echo "SU2_CFD not on PATH -- check SU2_RUN"; exit 1; }
SU2_CFD --help >/dev/null 2>&1 || { echo "SU2_CFD not runnable -- check LD_LIBRARY_PATH / OPENMPI_LIB"; exit 1; }
command -v python >/dev/null 2>&1 || { echo "python not on PATH"; exit 1; }
python -c "import mpi4py" 2>/dev/null || echo "WARNING: mpi4py not importable in this python -- shape_optimization.py needs it"

python "$SU2_RUN/shape_optimization.py" -f opt.cfg -n "$NP" 2>&1 | tee opt.log
