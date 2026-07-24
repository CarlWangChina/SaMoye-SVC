#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_DIR="${ROOT_DIR}/SaMoye-Model"
PYTHON="${SAMOYE_PYTHON:-python3}"
EXP_NAME="sovits_spk_expand_v2_mix"
TRAIN_LOG="train_v2_mix.log"

cd "${MODEL_DIR}"

# Verify data lists exist
if [[ ! -f files/train_expand_v2_mix.txt || ! -f files/valid_expand_v2_mix.txt ]]; then
    echo "[v2] ERROR: missing training list files. Run run_v2_full_pipeline.sh first."
    exit 1
fi

echo "[v2] Training list stats:"
wc -l files/train_expand_v2_mix.txt files/valid_expand_v2_mix.txt

V2_NEW_COUNT=$(grep -c 'data_expand_v2_new/' files/train_expand_v2_mix.txt || true)
if [[ "${V2_NEW_COUNT}" -eq 0 ]]; then
    echo "[v2] ERROR: files/train_expand_v2_mix.txt contains no data_expand_v2_new samples."
    echo "[v2] Run v2 preprocessing and rebuild the mix list before starting v2 training."
    exit 1
fi
echo "[v2] v2-new samples in train list: ${V2_NEW_COUNT}"

echo "[v2] Pretrain checkpoint:"
ls -lh chkpt/sovits_spk_expand_v1_core/sovits_spk_expand_v1_core_0019.pt

if ! nvidia-smi -L >/tmp/samoye_nvidia_smi.txt 2>/tmp/samoye_nvidia_smi.err; then
    echo "[v2] ERROR: GPU is not available on this machine."
    cat /tmp/samoye_nvidia_smi.err
    exit 1
fi
echo "[v2] GPUs available: $(wc -l < /tmp/samoye_nvidia_smi.txt)"
echo "[v2] Starting v2-mix training..."
echo ""

${PYTHON} svc_trainer.py \
    -c configs/sovits_spk_expand_v2_mix.yaml \
    -n "${EXP_NAME}" \
    2>&1 | tee "${TRAIN_LOG}"

if ! ${PYTHON} ../scripts/plot_training_metrics.py \
    --run-name "${EXP_NAME}" \
    --model-dir . \
    --log "${TRAIN_LOG}" \
    --output-dir ../outputs/training_plots; then
    echo "[v2] WARNING: failed to generate training metric plots."
fi
