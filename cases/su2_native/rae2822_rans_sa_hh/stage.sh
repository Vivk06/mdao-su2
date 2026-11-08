#!/usr/bin/env bash
# Stage mesh before tarring for transport; avoid a duplicate in source.
set -euo pipefail
cd "$(dirname "$0")"
SRC=../../rae2822_rans_sa_hh/input/mesh/mesh_RAE2822_turb.su2
if [ ! -f "$SRC" ]; then
    echo "ERROR: expected mesh at $SRC -- run this from inside the repo checkout."
    exit 1
fi
cp "$SRC" mesh_RAE2822_turb.su2
echo "staged $(du -h mesh_RAE2822_turb.su2 | cut -f1) -> mesh_RAE2822_turb.su2"
