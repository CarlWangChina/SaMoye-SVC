#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_DIR="${ROOT_DIR}/SaMoye-Model"
DATA_DIR="${ROOT_DIR}/data_samoye_expand_v1"
PYTHON="${SAMOYE_PYTHON:-python3}"

echo "[v1] ROOT_DIR=${ROOT_DIR}"
echo "[v1] MODEL_DIR=${MODEL_DIR}"
echo "[v1] DATA_DIR=${DATA_DIR}"

cd "${MODEL_DIR}"

# 1) spec from 32k wave
${PYTHON} prepare/preprocess_spec.py -w "${DATA_DIR}/waves-32k" -s "${DATA_DIR}/specs" -t 0

# 2) f0/hubert/ppg/speaker from 16k wave
${PYTHON} prepare/preprocess_f0.py -w "${DATA_DIR}/waves-16k" -p "${DATA_DIR}/pitch" -t 0
${PYTHON} prepare/preprocess_hubert.py -w "${DATA_DIR}/waves-16k" -v "${DATA_DIR}/hubert"
${PYTHON} prepare/preprocess_ppg.py -w "${DATA_DIR}/waves-16k" -p "${DATA_DIR}/whisper"
${PYTHON} prepare/preprocess_speaker.py "${DATA_DIR}/waves-16k" "${DATA_DIR}/speaker" -t 0

# 3) train/valid list
${PYTHON} prepare/preprocess_train.py --src "${DATA_DIR}"

echo "[v1] feature pipeline finished."
