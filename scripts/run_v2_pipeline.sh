#!/bin/bash
set -e
PYTHON="${SAMOYE_PYTHON:-python3}"
V2_DIR="data_expand_v2_new"
echo "=== SaMoye v2-expand Pipeline ==="
echo "[1/6] Extracting spectrograms..."
$PYTHON prepare/preprocess_spec.py -w ${V2_DIR}/waves-32k -s ${V2_DIR}/specs -t 0
echo "[2/6] Extracting F0 (pitch)..."
$PYTHON prepare/preprocess_f0.py -w ${V2_DIR}/waves-16k -p ${V2_DIR}/pitch -t 0
echo "[3/6] Extracting HuBERT features..."
$PYTHON prepare/preprocess_hubert.py -w ${V2_DIR}/waves-16k -v ${V2_DIR}/hubert
echo "[4/6] Extracting Whisper PPG features..."
$PYTHON prepare/preprocess_ppg.py -w ${V2_DIR}/waves-16k -p ${V2_DIR}/whisper
echo "[5/6] Extracting speaker embeddings..."
$PYTHON prepare/preprocess_speaker.py ${V2_DIR}/waves-16k ${V2_DIR}/speaker -t 0
echo "[6/6] Building merged train list..."
$PYTHON ../scripts/05_build_v2_mix_trainlist.py
echo "=== Pipeline Complete! ==="
echo "Start training: $PYTHON svc_trainer.py -c configs/sovits_spk_expand_v2_mix.yaml -n sovits_spk_expand_v2_mix"
