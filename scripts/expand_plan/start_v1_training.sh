#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_DIR="${ROOT_DIR}/SaMoye-Model"
PYTHON="${SAMOYE_PYTHON:-python3}"
EXP_NAME="sovits_spk_expand_v1_core"
TRAIN_LOG="train_v1_core.log"

cd "${MODEL_DIR}"
mkdir -p files

if [[ ! -f files/train.txt || ! -f files/valid.txt ]]; then
  echo "[v1] missing files/train.txt or files/valid.txt. Run feature pipeline first."
  exit 1
fi

cp files/train.txt files/train_expand_v1_core.txt
cp files/valid.txt files/valid_expand_v1_core.txt

echo "[v1] training list prepared:"
wc -l files/train_expand_v1_core.txt files/valid_expand_v1_core.txt

${PYTHON} svc_trainer.py \
    -c configs/sovits_spk_expand_v1_core.yaml \
    -n "${EXP_NAME}" \
    2>&1 | tee "${TRAIN_LOG}"

if ! ${PYTHON} ../scripts/plot_training_metrics.py \
    --run-name "${EXP_NAME}" \
    --model-dir . \
    --log "${TRAIN_LOG}" \
    --output-dir ../outputs/training_plots; then
    echo "[v1] WARNING: failed to generate training metric plots."
fi
