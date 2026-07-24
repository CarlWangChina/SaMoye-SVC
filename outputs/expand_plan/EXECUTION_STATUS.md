# SaMoye Expand Plan - Execution Status

## Current State (2026-05-01)

### Completed

- v1-core feature data exists under `SaMoye-Model/data_expand_v1`:
  - `waves-16k`: 26221 files
  - `waves-32k`: 26221 files
  - `specs`: 26221 files
  - `pitch`: 26221 files
  - `hubert`: 26221 files
  - `whisper`: 26221 files
  - `speaker`: 26205 files
- v1-core train/valid lists exist:
  - `SaMoye-Model/files/train_expand_v1_core.txt`: 26195 lines
  - `SaMoye-Model/files/valid_expand_v1_core.txt`: 10 lines
- v1-core training completed through epoch 19:
  - `SaMoye-Model/chkpt/sovits_spk_expand_v1_core/sovits_spk_expand_v1_core_0019.pt`
  - latest log: `SaMoye-Model/logs/sovits_spk_expand_v1_core/sovits_spk_expand_v1_core-1773743072.log`

### Not Ready / Blockers

- `SaMoye-Model/data_expand_v2_new` does not exist, so v2-expand data has not been materialized or feature-extracted.
- `SaMoye-Model/files/train_expand_v2_mix.txt` currently has the same line count as v1 and contains no verified v2-new samples; do not treat it as a real v2 mix list until rebuilt.
- The old `/tmp/samoye_mamba/envs/samoye/bin/python*` runtime path no longer exists.
- The default `python3` currently lacks required dependencies such as `torch`, `librosa`, `pyworld`, `omegaconf`, `numpy`, and `soundfile`.
- `nvidia-smi` currently fails, so GPU feature extraction/training cannot run on this machine/session until GPU drivers are available.

## Next Steps

1. Restore or create a Python runtime with SaMoye dependencies, then run scripts with `SAMOYE_PYTHON=/path/to/python`.
2. Put the original raw `train-data` directory at project root (`./train-data`) or adjust the scripts to the actual raw-data location.
3. Build v2-new raw segments:
   - `SAMOYE_PYTHON=/path/to/python python scripts/03_flatten_and_cut_v2.py`
4. Extract v2-new features:
   - `cd SaMoye-Model`
   - `SAMOYE_PYTHON=/path/to/python bash ../scripts/04_preprocess_and_train_v2.sh`
5. Start v2 training only after `files/train_expand_v2_mix.txt` contains `data_expand_v2_new/` entries and GPU is available:
   - `SAMOYE_PYTHON=/path/to/python bash ../scripts/expand_plan/start_v2_training.sh`

## Update (2026-05-26)

- Runtime restored/found: `/data0/backup/w-4090/micromamba/envs/samoye/bin/python`.
- GPU is available outside the sandbox: 8x RTX 4090.
- v2-new raw segmentation was generated from `歌曲录音素材单声道降噪去80hz以下版` with `NUMBA_CACHE_DIR=/tmp/numba_cache`.
- Filtering before segmentation:
  - input after filters: 1075 wav
  - skipped label `异性/朋友/亲友`: 602
  - skipped sample rate `<44100`: 30
- Segmentation output:
  - script-reported segments: 30007
  - script-reported speakers: 1008
- Active background session:
  - `screen`: `samoye_v2_prepare_train`
  - wrapper: `scripts/expand_plan/run_v2_prepare_and_train_2026_05_26.sh`
  - log: `outputs/expand_plan/v2_prepare_train_2026-05-26.log`
- The wrapper is running resampling, 8 GPU feature extraction, v2 mix-list building, then `start_v2_training.sh`.
- `run_v2_feature_pipeline.sh` and `run_v2_full_pipeline.sh` were fixed on 2026-05-26 to use relative `data_expand_v2_new` paths because the legacy preprocess scripts test `./${path}` internally and skip absolute paths.

## Monitor Snapshot (2026-05-26 20:22 CST)

- `screen` session `1610538.samoye_v2_prepare_train` is still detached and running.
- Current log tail is still in the v2 feature-preparation stage, processing `spec` and `f0` speaker batches.
- GPU status: all 8 RTX 4090 cards are idle at 0% utilization and 3 MiB memory, which is expected before the HuBERT/Whisper GPU stages start.
- Current feature counts under `SaMoye-Model/data_expand_v2_new`:
  - `specs`: 7108
  - `pitch`: 4057
  - `hubert`: 0
  - `whisper`: 0
  - `speaker`: 11437
- `SaMoye-Model/files/train_expand_v2_mix.txt` still contains 0 `data_expand_v2_new/` entries, so v2 mix-list rebuild and model training have not started yet.
- No new v2 checkpoint exists yet; latest checkpoint is still `SaMoye-Model/chkpt/sovits_spk_expand_v1_core/sovits_spk_expand_v1_core_0019.pt`.

## ETA Snapshot (2026-05-26 20:27 CST)

- Expected v2-new sample count: 29195.
- Current Phase 1 progress:
  - `specs`: 12099/29195
  - `pitch`: 6845/29195
  - `hubert`: 0/29195
  - `whisper`: 0/29195
- Current bottleneck is `pitch/f0`, observed at roughly 599 files/minute since the 20:15 restart.
- Estimated Phase 1 completion: around 21:05 CST if the current rate holds.
- Estimated remaining feature-preparation time before v2 training starts: roughly 1.5 to 2 hours, depending mostly on the subsequent 8-GPU HuBERT and Whisper stages.

## Monitor Snapshot (2026-05-26 20:56 CST)

- `screen` session `1610538.samoye_v2_prepare_train` is still detached and running.
- No error keywords were found in the current log scan (`Traceback`, `RuntimeError`, `Killed`, `CUDA out`, etc.).
- Current phase: still Phase 1 CPU-bound feature extraction, now only `f0/pitch` is running.
- GPU status: all 8 RTX 4090 cards are still idle at 0% utilization and 3 MiB memory; HuBERT/Whisper GPU stages have not started yet.
- Current v2-new feature counts:
  - `waves-16k`: 29195/29195
  - `specs`: 29195/29195 complete
  - `pitch`: 25793/29195
  - `hubert`: 0/29195
  - `whisper`: 0/29195
  - `speaker`: 11437 existing from the earlier partial run; final speaker extraction has not rerun in this 8-GPU pipeline yet.
- `files/train_expand_v2_mix.txt` still contains 0 `data_expand_v2_new/` entries, so v2 mix-list rebuild and model training have not started yet.
- No v2 checkpoint exists yet; latest checkpoint is still `SaMoye-Model/chkpt/sovits_spk_expand_v1_core/sovits_spk_expand_v1_core_0019.pt`.

## Monitor Snapshot (2026-05-26 22:52 CST)

- `screen` session `1610538.samoye_v2_prepare_train` is still detached and running.
- Data feature preparation is complete:
  - `waves-16k`: 29195/29195
  - `specs`: 29195/29195
  - `pitch`: 29195/29195
  - `hubert`: 29195/29195
  - `whisper`: 29195/29195
  - `speaker`: 29195/29195
- v2 mix train list was rebuilt:
  - total train samples: 40300
  - v2-new train samples: 14105
  - v2-new ratio: 35.0%
  - valid samples: 10
- v2 training started at 2026-05-26 21:55 CST with config `configs/sovits_spk_expand_v2_mix.yaml`.
- Current config: 20 epochs, batch size 16, pretrain `chkpt/sovits_spk_expand_v1_core/sovits_spk_expand_v1_core_0019.pt`, save interval 1 epoch.
- Latest training log line at 2026-05-26 22:52 CST:
  - epoch 2, step 502
  - approximately 59% through epoch 2, based on 315 steps/epoch.
- Current v2 checkpoint:
  - `SaMoye-Model/chkpt/sovits_spk_expand_v2_mix/sovits_spk_expand_v2_mix_0001.pt`
- GPU memory indicates training is active on all 8 RTX 4090 cards, with about 18-21 GiB used per card.
- No error keywords were found in the latest v2 training log scan.
