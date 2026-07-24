#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_DIR="${ROOT_DIR}/SaMoye-Model"
DATA_DIR="data_expand_v2_new"
SCRIPTS_DIR="${ROOT_DIR}/scripts/expand_plan"
PYTHON="${SAMOYE_PYTHON:-python3}"
NUM_GPUS=8

echo "============================================"
echo "[v2] SaMoye v2-mix Feature Pipeline (8 GPU)"
echo "============================================"
echo "[v2] ROOT_DIR=${ROOT_DIR}"
echo "[v2] MODEL_DIR=${MODEL_DIR}"
echo "[v2] DATA_DIR=${MODEL_DIR}/${DATA_DIR}"
echo ""

cd "${MODEL_DIR}"

# ====== Phase 1: CPU-bound features (spec + f0) in parallel ======
echo "[v2] Phase 1: CPU-bound features (spec + f0)"

echo "[v2] 1a: Extracting specs from waves-32k..."
${PYTHON} prepare/preprocess_spec.py \
    -w "${DATA_DIR}/waves-32k" \
    -s "${DATA_DIR}/specs" \
    -t 0 &
PID_SPEC=$!

echo "[v2] 1b: Extracting pitch (f0) from waves-16k..."
${PYTHON} prepare/preprocess_f0.py \
    -w "${DATA_DIR}/waves-16k" \
    -p "${DATA_DIR}/pitch" \
    -t 0 &
PID_F0=$!

echo "[v2] Waiting for spec and f0 extraction..."
wait ${PID_SPEC}
echo "[v2] Spec extraction done."
wait ${PID_F0}
echo "[v2] F0 extraction done."

# ====== Phase 2: GPU-bound features (hubert + ppg) across 8 GPUs ======
echo ""
echo "[v2] Phase 2: GPU-bound features (hubert on 8 GPUs)"
${PYTHON} "${SCRIPTS_DIR}/parallel_hubert.py" \
    --wav-dir "${DATA_DIR}/waves-16k" \
    --vec-dir "${DATA_DIR}/hubert" \
    --num-gpus ${NUM_GPUS}

echo ""
echo "[v2] Phase 3: GPU-bound features (whisper PPG on 8 GPUs)"
${PYTHON} "${SCRIPTS_DIR}/parallel_ppg.py" \
    --wav-dir "${DATA_DIR}/waves-16k" \
    --ppg-dir "${DATA_DIR}/whisper" \
    --num-gpus ${NUM_GPUS}

# ====== Phase 4: Speaker embeddings ======
echo ""
echo "[v2] Phase 4: Speaker embeddings"
${PYTHON} prepare/preprocess_speaker.py \
    "${DATA_DIR}/waves-16k" \
    "${DATA_DIR}/speaker" \
    -t 0

# ====== Phase 5: Generate training list ======
echo ""
echo "[v2] Phase 5: Generate training list"
${PYTHON} prepare/preprocess_train.py --src "${DATA_DIR}"

# Rename to v2_mix names
cp files/train.txt files/train_expand_v2_mix.txt
cp files/valid.txt files/valid_expand_v2_mix.txt

echo ""
echo "[v2] Training list ready:"
wc -l files/train_expand_v2_mix.txt files/valid_expand_v2_mix.txt

echo ""
echo "============================================"
echo "[v2] Feature pipeline complete!"
echo "============================================"
echo ""
echo "To start training:"
echo "  cd ${MODEL_DIR}"
echo "  ${PYTHON} svc_trainer.py -c configs/sovits_spk_expand_v2_mix.yaml -n sovits_spk_expand_v2_mix"
