#!/bin/bash
# Step 2: Resample, extract features, generate train list, and start training.
# Run from SaMoye-Model directory.
#
# Usage:
#   cd /path/to/SaMoye-SVC/SaMoye-Model
#   bash ../scripts/02_preprocess_and_train.sh

set -e

PYTHON="${SAMOYE_PYTHON:-python3}"
DATA_DIR="data_expand_v1"
CONFIG="configs/sovits_spk_expand_v1_core.yaml"
EXP_NAME="sovits_spk_expand_v1_core"

echo "============================================"
echo "  SaMoye Expand v1 - Preprocessing Pipeline"
echo "============================================"

# --- Step 2a: Resample to 32k and 16k ---
echo ""
echo "[1/7] Resampling to 32kHz..."
$PYTHON prepare/preprocess_a.py -w ${DATA_DIR}/raw_clean -o ${DATA_DIR}/waves-32k -s 32000 -t 0

echo "[2/7] Resampling to 16kHz..."
$PYTHON prepare/preprocess_a.py -w ${DATA_DIR}/raw_clean -o ${DATA_DIR}/waves-16k -s 16000 -t 0

# --- Step 2b: Feature extraction ---
echo ""
echo "[3/7] Extracting spectrograms..."
$PYTHON prepare/preprocess_spec.py -w ${DATA_DIR}/waves-32k -s ${DATA_DIR}/specs -t 0

echo "[4/7] Extracting F0 (pitch)..."
$PYTHON prepare/preprocess_f0.py -w ${DATA_DIR}/waves-16k -p ${DATA_DIR}/pitch -t 0

echo "[5/7] Extracting HuBERT features..."
$PYTHON prepare/preprocess_hubert.py -w ${DATA_DIR}/waves-16k -v ${DATA_DIR}/hubert

echo "[6/7] Extracting Whisper PPG features..."
$PYTHON prepare/preprocess_ppg.py -w ${DATA_DIR}/waves-16k -p ${DATA_DIR}/whisper

echo "[6.5/7] Extracting speaker embeddings..."
$PYTHON prepare/preprocess_speaker.py ${DATA_DIR}/waves-16k ${DATA_DIR}/speaker -t 0

# --- Step 2c: Generate train/valid lists ---
echo ""
echo "[7/7] Generating train/valid file lists..."
$PYTHON prepare/preprocess_train.py --src ${DATA_DIR}

# Rename generated files to match config expectations
mkdir -p files
if [ -f files/train.txt ]; then
    cp files/train.txt files/train_expand_v1_core.txt
    echo "  -> files/train_expand_v1_core.txt created"
fi
if [ -f files/valid.txt ]; then
    cp files/valid.txt files/valid_expand_v1_core.txt
    echo "  -> files/valid_expand_v1_core.txt created"
fi

echo ""
echo "============================================"
echo "  Preprocessing complete!"
echo "============================================"

TRAIN_COUNT=$(wc -l < files/train_expand_v1_core.txt 2>/dev/null || echo "0")
VALID_COUNT=$(wc -l < files/valid_expand_v1_core.txt 2>/dev/null || echo "0")
echo "  Train samples: ${TRAIN_COUNT}"
echo "  Valid samples: ${VALID_COUNT}"
echo ""
echo "To start training:"
echo "  cd $(pwd)"
echo "  $PYTHON svc_trainer.py -c ${CONFIG} -n ${EXP_NAME}"
echo ""
echo "To monitor with Tensorboard:"
echo "  tensorboard --logdir=logs/${EXP_NAME} --port 12345"
