# SaMoye-SVC — expand_train_jxb

本分支增加了约 4000 名中国歌声、说话人数据进行扩大训练。

## Expanded training

```bash
cd SaMoye-Model

# 第一阶段扩大训练
python svc_trainer.py \
  -c configs/sovits_spk_expand_v1_core.yaml \
  -n sovits_spk_expand_v1_core

# 第一阶段继续训练
python svc_trainer.py \
  -c configs/sovits_spk_expand_v1_core_continue.yaml \
  -n sovits_spk_expand_v1_core_continue_e60_20260610

# 第二阶段混合训练
python svc_trainer.py \
  -c configs/sovits_spk_expand_v2_mix.yaml \
  -n sovits_spk_expand_v2_mix
```

## Model weights

Hugging Face 的几个权重 model 链接如下：

- **扩大训练验证效果最佳**：[sovits_spk_expand_v2_mix_0015.pt](https://huggingface.co/karl-wang/SaMoye-SVC-expand-train-jxb/resolve/main/checkpoints/sovits_spk_expand_v2_mix/sovits_spk_expand_v2_mix_0015.pt?download=true)
- **最近保存的 checkpoint**：[sovits_spk_expand_v1_core_continue_e60_20260610_0031.pt](https://huggingface.co/karl-wang/SaMoye-SVC-expand-train-jxb/resolve/main/checkpoints/sovits_spk_expand_v1_core_continue_e60_20260610/sovits_spk_expand_v1_core_continue_e60_20260610_0031.pt?download=true)
- **后续训练依赖的父 checkpoint**：[sovits_spk_expand_v1_core_0019.pt](https://huggingface.co/karl-wang/SaMoye-SVC-expand-train-jxb/resolve/main/checkpoints/sovits_spk_expand_v1_core/sovits_spk_expand_v1_core_0019.pt?download=true)

下载后需要分别放在以下路径：

```text
SaMoye-Model/chkpt/sovits_spk_expand_v2_mix/sovits_spk_expand_v2_mix_0015.pt
SaMoye-Model/chkpt/sovits_spk_expand_v1_core_continue_e60_20260610/sovits_spk_expand_v1_core_continue_e60_20260610_0031.pt
SaMoye-Model/chkpt/sovits_spk_expand_v1_core/sovits_spk_expand_v1_core_0019.pt
```

`sovits_spk_expand_v2_mix_0015.pt` 在 TensorBoard step 5040 取得三组扩大训练记录中的最低验证 mel loss（`0.0590887`），因此作为当前验证效果最佳权重。`0031.pt` 是最近一次完整保存的 checkpoint；`v1_core_0019.pt` 是两个后续训练配置共同依赖的父 checkpoint。其他中间或废弃 checkpoint 不上传。

## TensorBoard

```bash
tensorboard --logdir=logs --port 12345
```

## Inference

```bash
python svc_inference.py \
  --config configs/sovits_spk_expand_v1_core_continue.yaml \
  --model chkpt/sovits_spk_expand_v1_core_continue_e60_20260610/sovits_spk_expand_v1_core_continue_e60_20260610_0031.pt \
  --spk spk.wav \
  --wave content.wav
```
