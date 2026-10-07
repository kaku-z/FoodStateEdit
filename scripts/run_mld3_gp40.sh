#!/usr/bin/env bash
set -e
root="$(cd -- "$(dirname -- "$0")/.." && pwd)"
export OMP_NUM_THREADS=4
export PYOPENGL_PLATFORM=egl
exec /host/space0/guo-z/tf-ufi/food3d_pilot_20260928/venv/bin/python -u \
  "$root/scripts/run_mld3.py" --checkpoint "$root/weights/mld2_source65k.pt" "$@"
