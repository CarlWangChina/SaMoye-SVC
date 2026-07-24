#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_DIR="${ROOT_DIR}/SaMoye-Model"
PYTHON="${SAMOYE_PYTHON:-/data0/backup/w-4090/micromamba/envs/samoye/bin/python}"
LOG_DIR="${ROOT_DIR}/outputs/expand_plan"
LOG_FILE="${LOG_DIR}/v2_prepare_train_2026-05-26.log"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "============================================"
echo "[v2-run] SaMoye v2 prepare + train"
echo "[v2-run] started: $(date '+%F %T %Z')"
echo "[v2-run] ROOT_DIR=${ROOT_DIR}"
echo "[v2-run] MODEL_DIR=${MODEL_DIR}"
echo "[v2-run] PYTHON=${PYTHON}"
echo "============================================"

export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export SAMOYE_PYTHON="${PYTHON}"

cd "${MODEL_DIR}"
RAW_COUNT="$(find data_expand_v2_new/raw_clean -type f -name '*.wav' | wc -l)"

echo "[v2-run] Step 1: resample v2-new to 32kHz"
W32_COUNT="$(find data_expand_v2_new/waves-32k -type f -name '*.wav' 2>/dev/null | wc -l || true)"
if [[ "${W32_COUNT}" == "${RAW_COUNT}" ]]; then
  echo "[v2-run] 32kHz resample already complete: ${W32_COUNT}/${RAW_COUNT}"
else
  "${PYTHON}" prepare/preprocess_a.py \
    -w data_expand_v2_new/raw_clean \
    -o data_expand_v2_new/waves-32k \
    -s 32000 \
    -t 0
fi

echo "[v2-run] Step 2: resample v2-new to 16kHz"
W16_COUNT="$(find data_expand_v2_new/waves-16k -type f -name '*.wav' 2>/dev/null | wc -l || true)"
if [[ "${W16_COUNT}" == "${RAW_COUNT}" ]]; then
  echo "[v2-run] 16kHz resample already complete: ${W16_COUNT}/${RAW_COUNT}"
else
  "${PYTHON}" prepare/preprocess_a.py \
    -w data_expand_v2_new/raw_clean \
    -o data_expand_v2_new/waves-16k \
    -s 16000 \
    -t 0
fi

echo "[v2-run] Step 3: extract features with 8-GPU v2 pipeline"
bash ../scripts/expand_plan/run_v2_full_pipeline.sh

echo "[v2-run] Step 4: rebuild v2-mix train list with capped new-data ratio"
"${PYTHON}" ../scripts/05_build_v2_mix_trainlist.py

echo "[v2-run] Step 5: show final list stats"
wc -l files/train_expand_v2_mix.txt files/valid_expand_v2_mix.txt
grep -c 'data_expand_v2_new/' files/train_expand_v2_mix.txt || true

echo "[v2-run] Step 6: start v2 training"
bash ../scripts/expand_plan/start_v2_training.sh

echo "[v2-run] finished: $(date '+%F %T %Z')"
