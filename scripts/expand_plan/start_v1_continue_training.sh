#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_DIR="${ROOT_DIR}/SaMoye-Model"
PYTHON="${SAMOYE_PYTHON:-python3}"
EXP_NAME="${SAMOYE_EXP_NAME:-sovits_spk_expand_v1_core_continue}"
CONFIG="${SAMOYE_CONFIG:-configs/sovits_spk_expand_v1_core_continue.yaml}"
CHECKPOINT="${SAMOYE_CHECKPOINT:-chkpt/sovits_spk_expand_v1_core/sovits_spk_expand_v1_core_0019.pt}"
TRAIN_LOG="${SAMOYE_TRAIN_LOG:-train_v1_core_continue.log}"

cd "${MODEL_DIR}"

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "[v1-continue] ERROR: missing checkpoint: ${CHECKPOINT}"
    exit 1
fi

if [[ ! -f files/train_expand_v1_core.txt || ! -f files/valid_expand_v1_core.txt ]]; then
    echo "[v1-continue] ERROR: missing v1-core train/valid lists."
    exit 1
fi

echo "[v1-continue] Training list stats:"
wc -l files/train_expand_v1_core.txt files/valid_expand_v1_core.txt

echo "[v1-continue] Checkpoint:"
ls -lh "${CHECKPOINT}"

if ! nvidia-smi -L >/tmp/samoye_nvidia_smi.txt 2>/tmp/samoye_nvidia_smi.err; then
    echo "[v1-continue] ERROR: GPU is not available on this machine."
    cat /tmp/samoye_nvidia_smi.err
    exit 1
fi

echo "[v1-continue] GPUs available: $(wc -l < /tmp/samoye_nvidia_smi.txt)"
echo "[v1-continue] Starting training with ${PYTHON}"

${PYTHON} svc_trainer.py \
    -c "${CONFIG}" \
    -n "${EXP_NAME}" \
    -p "${CHECKPOINT}" \
    2>&1 | tee "${TRAIN_LOG}"

if ! ${PYTHON} ../scripts/plot_training_metrics.py \
    --run-name "${EXP_NAME}" \
    --model-dir . \
    --log "${TRAIN_LOG}" \
    --output-dir ../outputs/training_plots; then
    echo "[v1-continue] WARNING: failed to generate training metric plots."
fi
