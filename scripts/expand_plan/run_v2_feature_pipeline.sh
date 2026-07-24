#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_DIR="${ROOT_DIR}/SaMoye-Model"
DATA_DIR="data_expand_v2_new"
PYTHON="${SAMOYE_PYTHON:-python3}"

echo "[v2] ROOT_DIR=${ROOT_DIR}"
echo "[v2] MODEL_DIR=${MODEL_DIR}"
echo "[v2] DATA_DIR=${MODEL_DIR}/${DATA_DIR}"
echo "[v2] Using python: ${PYTHON}"

cd "${MODEL_DIR}"

# 1) spec from 32k wave (will skip already-extracted files if preprocess_spec.py supports it)
echo "[v2] Step 1/5: Extracting specs..."
${PYTHON} prepare/preprocess_spec.py -w "${DATA_DIR}/waves-32k" -s "${DATA_DIR}/specs" -t 0

# 2) f0 pitch from 16k wave
echo "[v2] Step 2/5: Extracting pitch (f0)..."
${PYTHON} prepare/preprocess_f0.py -w "${DATA_DIR}/waves-16k" -p "${DATA_DIR}/pitch" -t 0

# 3) hubert from 16k wave
echo "[v2] Step 3/5: Extracting hubert..."
${PYTHON} prepare/preprocess_hubert.py -w "${DATA_DIR}/waves-16k" -v "${DATA_DIR}/hubert"

# 4) whisper ppg from 16k wave
echo "[v2] Step 4/5: Extracting whisper ppg..."
${PYTHON} prepare/preprocess_ppg.py -w "${DATA_DIR}/waves-16k" -p "${DATA_DIR}/whisper"

# 5) speaker embeddings from 16k wave
echo "[v2] Step 5/5: Extracting speaker embeddings..."
${PYTHON} prepare/preprocess_speaker.py "${DATA_DIR}/waves-16k" "${DATA_DIR}/speaker" -t 0

# 6) generate train/valid list
echo "[v2] Step 6: Generating train/valid lists..."
${PYTHON} prepare/preprocess_train.py --src "${DATA_DIR}"

echo "[v2] Feature pipeline finished!"
echo "[v2] Checking generated files..."
wc -l "${MODEL_DIR}/files/train.txt" "${MODEL_DIR}/files/valid.txt" 2>/dev/null || true
