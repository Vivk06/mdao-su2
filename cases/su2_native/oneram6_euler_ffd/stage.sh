#!/usr/bin/env bash
# Stage mesh before tarring for transport; avoid a duplicate in source.
set -euo pipefail
cd "$(dirname "$0")"
SRC=../../oneram6_euler_ffd/input/mesh/mesh_ONERAM6_inv_FFD.su2
if [ ! -f "$SRC" ]; then
    echo "ERROR: expected mesh at $SRC -- run this from inside the repo checkout."
    exit 1
fi
cp "$SRC" mesh_ONERAM6_inv_FFD.su2
echo "staged $(du -h mesh_ONERAM6_inv_FFD.su2 | cut -f1) -> mesh_ONERAM6_inv_FFD.su2"
