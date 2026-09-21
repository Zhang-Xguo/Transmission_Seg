#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# Keep all deep-learning processes on physical GPU 4-6 and cap CPU threads.
export CUDA_VISIBLE_DEVICES=4,5,6
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2

PYTHON="${PYTHON:-python}"
OUT="exp/gridnethd/spunet_7class_public_stage_a"
mkdir -p "$OUT"
set +e
"$PYTHON" tools/train.py \
  --config-file configs/gridnethd/SpUNet_gridnethd_7class_public_stage_a.py \
  --num-gpus 3
STATUS=$?
set -e
"$PYTHON" scripts/report_spunet_stage_a.py \
  --experiment "$OUT" --exit-code "$STATUS"
exit "$STATUS"
