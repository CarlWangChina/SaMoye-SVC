#!/bin/bash
# Step 4: Preprocess v2 new data, merge with v1, and prepare for v2-mix training.
#
# This script:
#   1. Extracts features for v2-new data
#   2. Merges v1 + v2 data into combined train list
#   3. Creates v2-mix config pointing to v1-core epoch 19 checkpoint
#
# Usage:
#   cd /path/to/SaMoye-SVC/SaMoye-Model
#   bash ../scripts/04_preprocess_and_train_v2.sh

set -e

PYTHON="${SAMOYE_PYTHON:-python3}"
V2_NEW_DATA="data_expand_v2_new"
V1_DATA="data_expand_v1"
CONFIG="configs/sovits_spk_expand_v2_mix.yaml"
EXP_NAME="sovits_spk_expand_v2_mix"

echo "============================================"
echo "  SaMoye Expand v2 - Preprocessing Pipeline"
echo "============================================"

# --- Step 1: Feature extraction for v2-new data ---
echo ""
echo "[1/7] Resampling v2-new to 32kHz..."
$PYTHON prepare/preprocess_a.py -w ${V2_NEW_DATA}/raw_clean -o ${V2_NEW_DATA}/waves-32k -s 32000 -t 0

echo "[2/7] Resampling v2-new to 16kHz..."
$PYTHON prepare/preprocess_a.py -w ${V2_NEW_DATA}/raw_clean -o ${V2_NEW_DATA}/waves-16k -s 16000 -t 0

echo "[3/7] Extracting spectrograms..."
$PYTHON prepare/preprocess_spec.py -w ${V2_NEW_DATA}/waves-32k -s ${V2_NEW_DATA}/specs -t 0

echo "[4/7] Extracting F0 (pitch)..."
$PYTHON prepare/preprocess_f0.py -w ${V2_NEW_DATA}/waves-16k -p ${V2_NEW_DATA}/pitch -t 0

echo "[5/7] Extracting HuBERT features..."
$PYTHON prepare/preprocess_hubert.py -w ${V2_NEW_DATA}/waves-16k -v ${V2_NEW_DATA}/hubert

echo "[6/7] Extracting Whisper PPG features..."
$PYTHON prepare/preprocess_ppg.py -w ${V2_NEW_DATA}/waves-16k -p ${V2_NEW_DATA}/whisper

echo "[6.5/7] Extracting speaker embeddings..."
$PYTHON prepare/preprocess_speaker.py ${V2_NEW_DATA}/waves-16k ${V2_NEW_DATA}/speaker -t 0

# --- Step 2: Generate v2-new train list ---
echo ""
echo "[7/7] Generating v2-new train/valid file lists..."
$PYTHON prepare/preprocess_train.py --src ${V2_NEW_DATA}

# Backup v2-new list
mkdir -p files
if [ -f files/train.txt ]; then
    cp files/train.txt files/train_expand_v2_new_only.txt
    echo "  -> files/train_expand_v2_new_only.txt created"
fi

# --- Step 3: Merge v1 + v2-new into combined list ---
echo ""
echo "Merging v1 + v2 train lists..."

# v1-core train list
V1_TRAIN="files/train_expand_v1_core.txt"
V2_NEW_TRAIN="files/train_expand_v2_new_only.txt"
V2_MIX_TRAIN="files/train_expand_v2_mix.txt"
V2_MIX_VALID="files/valid_expand_v2_mix.txt"

if [ ! -f "$V1_TRAIN" ]; then
    echo "ERROR: v1 train list not found: $V1_TRAIN"
    exit 1
fi

if [ ! -f "$V2_NEW_TRAIN" ]; then
    echo "ERROR: v2-new train list not found: $V2_NEW_TRAIN"
    exit 1
fi

# Combine: all v1 + all v2-new
cat "$V1_TRAIN" "$V2_NEW_TRAIN" > "$V2_MIX_TRAIN"
echo "  -> $V2_MIX_TRAIN created"

# For validation, use existing v1 valid set (already curated)
cp files/valid_expand_v1_core.txt "$V2_MIX_VALID"
echo "  -> $V2_MIX_VALID created"

V1_COUNT=$(wc -l < "$V1_TRAIN")
V2_NEW_COUNT=$(wc -l < "$V2_NEW_TRAIN")
TOTAL_COUNT=$(wc -l < "$V2_MIX_TRAIN")
V2_RATIO=$(echo "scale=1; $V2_NEW_COUNT * 100 / $TOTAL_COUNT" | bc)

echo ""
echo "============================================"
echo "  Preprocessing complete!"
echo "============================================"
echo "  v1-core samples:  $V1_COUNT"
echo "  v2-new samples:   $V2_NEW_COUNT"
echo "  Combined total:   $TOTAL_COUNT"
echo "  v2-new ratio:     ${V2_RATIO}%"
echo ""

# Check ratio warning
if (( $(echo "$V2_RATIO > 40" | bc -l) )); then
    echo "  WARNING: v2-new ratio exceeds 40% recommended limit!"
    echo "  Consider reducing v2-new data or using sampling."
fi

echo ""
echo "To start training:"
echo "  cd $(pwd)"
echo "  $PYTHON svc_trainer.py -c ${CONFIG} -n ${EXP_NAME}"
echo ""
echo "To monitor with Tensorboard:"
echo "  tensorboard --logdir=logs/${EXP_NAME} --port 12345"
