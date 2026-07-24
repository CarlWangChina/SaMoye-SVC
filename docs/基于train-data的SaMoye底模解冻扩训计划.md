# 基于 `train-data` 的 SaMoye 底模解冻扩训计划

## 1. 结论先行

当前仓库已经具备继续训练 SaMoye-SVC 的核心代码，但不能直接拿 `train-data` 原始音频开跑。原因很明确：

- 训练入口依赖的是特征清单文件，而不是原始 wav 目录。
- 当前训练代码默认是全参数训练，不存在现成的“部分冻结”开关；要做“底模解冻扩训”，最直接的路径就是从现有 1700h 权重继续全量训练。
- `train-data` 里的音频是多批次原始采集数据，采样率、命名规范、数据来源和标签可信度都不一致，必须先做数据重构。

基于仓库现状，最稳妥的方案不是“一次性把 500 多小时全灌进去”，而是：

1. 先做一版高质量子集 `v1-core`，完成目录重构、采样率统一、特征提取和小规模续训验证。
2. 验证成功后，再做 `v2-expand`，逐步把非降噪、未知标签、异性/朋友类数据加进来。
3. 全程沿用当前仓库的训练范式，先不改模型结构，只改数据组织、训练清单和训练阶段。

## 2. 我读取到的仓库事实

### 2.1 文档侧

- `docs/README.md` 说明了训练入口是 `python svc_trainer.py -c configs/sovits_spk_1700h.yaml -n ...`。
- `docs/SaMoye-SVC-问题回答整理.docx` 和英文 Q&A 明确提到：
  - 预处理可以参考 `whisper-sovits-svc` 的流程。
  - 训练时 pitch 实际使用的是 `npy`。
  - 缺失的 `files/train_filtered_1700_withoutnan.txt` 可以用预处理后生成的 `train.txt` 代替。
  - 1700 解冻版本的 `spk_emb` 维度是 `256`。
- `docs/samoye.pdf` 本地无法直接解析，因为当前环境缺少 PDF 文本提取工具和 Python PDF 库；下面的计划主要依据 README、Q&A、代码和数据实际情况制定。

### 2.2 代码侧

- `SaMoye-Model/configs/sovits_spk_1700h.yaml` 的默认配置是从 `vits_pretrain/sovits5.0.pretrain.pth` 起训。
- `SaMoye-Model/configs/sovits_spk_1700h_finetune_sing.yaml` 和 `..._finetune_read.yaml` 的思路是从已有 `chkpt/...pt` 继续训练 20 个 epoch。
- `SaMoye-Model/svc_trainer.py` 只有两种启动方式：
  - 用 `train.pretrain` 初始化。
  - 用 `--checkpoint_path` 断点续训。
- `SaMoye-Model/vits_extend/train.py` 默认会更新生成器和判别器全部参数，当前代码没有现成的冻结/解冻参数分组逻辑。
- `SaMoye-Model/vits/data_utils.py` 说明训练数据每行需要 6 个路径：
  - `wav|spec|pitch|hubert_vec|whisper_ppg|spk`
- 数据加载时虽然读取了 `spk` 路径，但实际训练使用的是 `wav_path` 在线过 speaker encoder，因此：
  - `speaker/*.spk.npy` 仍然需要生成来通过 `preprocess_train.py` 校验。
  - 真正决定训练能否启动的是 `waves-32k/specs/pitch/hubert/whisper` 五套数据是否齐全。
  - 如果直接喂 10 分钟级整歌，speaker encoder 会在训练时反复读取整段 wav，这会成为扩训速度和显存外的最大瓶颈。

### 2.3 代码深度审读补充（基于逐行阅读源码）

#### 2.3.1 训练时的有效片段长度硬限制

`data_utils.py:30-31` 中的过滤逻辑：

```python
items_min = int(self.segment_size / self.hop_length * 4)   # = 8000/320*4  = 100 帧 ≈ 1 秒
items_max = int(self.segment_size / self.hop_length * 16)   # = 8000/320*16 = 400 帧 ≈ 4 秒
```

这意味着**每个训练样本实际只使用最多 4 秒的随机窗口**。即使音频切成 20-40 秒，DataLoader 也只会从中随机截取 1-4 秒进行训练。切段的主要意义在于：
- 大幅减少特征提取的磁盘和计算开销（10 分钟整歌的 spec/hubert/ppg 特征文件远大于 30 秒片段）。
- 避免 speaker encoder 在线编码过长的 wav（见下节）。
- 过滤掉前奏、尾奏、长静音等无效内容，提升有效训练帧占比。

#### 2.3.2 Speaker Encoder 在线计算是训练吞吐的关键瓶颈

`models.py:187` — `SynthesizerTrn.forward()` 中每个 batch 的每条样本都调用 `self.spk_encoder_helper.forward(wav_paths)`，内部逻辑是（`prepare/preprocess_speaker.py:105-120`）：

1. 用 `AudioProcessor.load_wav()` **逐条加载原始 wav 文件**。
2. 计算 mel spectrogram。
3. 过 LSTM speaker encoder 得到 256 维 embedding。

这个过程**不走预提取的 `.spk.npy`，而是每个 step 在线计算**。如果 wav 文件是 10 分钟整歌，每次 forward 都要读取和处理整段音频，I/O 和计算开销远超模型本身的 forward/backward。**这是必须切段的最直接技术理由。**

#### 2.3.3 Pretrain 模式与 Checkpoint 续训模式的关键区别

`train.py:115-148` 的加载逻辑：

| 模式 | 加载内容 | Optimizer 状态 | Epoch/Step |
|------|---------|---------------|------------|
| `pretrain`（无 `--checkpoint_path`） | 仅 `model_g` + `model_d` 权重 | **从零初始化** | epoch=1, step=0 |
| `--checkpoint_path` | `model_g` + `model_d` + `optim_g` + `optim_d` | **恢复历史动量** | 恢复到保存时的 epoch 和 step |

对于扩训，两种模式的选择影响如下：

- **推荐用 `--checkpoint_path`**：保留 AdamW 的一阶/二阶动量估计，续训初期更平滑，不容易出现 loss 突跳。但注意学习率会按 `ExponentialLR` 的 `last_epoch = init_epoch - 2` 继续衰减，即恢复后的实际学习率 = `lr_config × lr_decay^(restored_epoch - 1)`。如果原始训练已经跑了很多 epoch，恢复后的有效学习率可能已经极小，需要在 yaml 里适当调大 `learning_rate` 来补偿。
- **用 `pretrain` 模式**：Optimizer 从零开始，学习率从配置值出发，更可控。但初始阶段可能因为动量缺失而出现短暂不稳定。适合"想要完全控制学习率策略"的场景。

#### 2.3.4 Finetune 配置已有参考范式

`sovits_spk_1700h_finetune_sing.yaml` 提供了一个已验证的扩训范式：

- `epochs: 20`
- `pretrain` 指向 `./chkpt/sovits_spk_1700h/sovits_spk_1700h_0012.pt`（用 pretrain 模式加载 checkpoint）
- `learning_rate: 5e-5`（与底模训练一致，因为 optimizer 从零开始）
- `log.save_interval` / `eval_interval` / `info_interval` 均设为 `1`（方便逐 epoch 监控）

这说明仓库作者此前的扩训做法是：**通过 `pretrain` 字段加载已有 checkpoint，optimizer 重新初始化，跑少量 epoch**。本次扩训可以沿用这个模式。

#### 2.3.5 训练损失函数构成

`train.py:222-268` 中，生成器的总损失由 7 项加权求和：

```
loss_g = score_loss + feat_loss + mel_loss + stft_loss + loss_kl_f + loss_kl_r * 0.5 + spk_loss * 2
```

各项含义：
- `mel_loss`：L1 mel spectrogram 重建损失（权重 `c_mel=1.0`）
- `stft_loss`：多分辨率 STFT 损失（权重 `c_stft=9`）
- `score_loss` / `feat_loss`：GAN 判别器对抗损失和特征匹配损失
- `loss_kl_f` / `loss_kl_r`：前向/反向 flow 的 KL 散度（权重 `c_kl=0.2`）
- `spk_loss`：说话人分类器的 cosine embedding 损失（权重 `×2`）

扩训时重点关注 `mel_loss` 和 `stft_loss` 是否稳定下降，`spk_loss` 是否维持在合理范围。如果 `spk_loss` 持续上升，说明新数据的说话人分布与底模差异过大，需要检查数据质量或降低学习率。

#### 2.3.6 梯度累积和学习率关系

`train.py:126-137` 中，判别器的学习率被设为 `learning_rate / accum_step`。当前配置 `accum_step=1`，所以生成器和判别器使用相同学习率。如果扩训时增大 `accum_step` 来模拟更大 batch，需注意判别器的学习率会同步降低。

#### 2.3.7 数据集各目录的文件数量分布

```
680  recordings_single_channel_denoised_cut80hz     ← 第一优先级
446  25年补充音频3
410  歌曲录音素材单声道降噪去80hz以下版              ← 第一优先级
192  25年6月至12月歌曲录音素材-门店端
130  25年4月歌曲录音素材-门店端130
 98  25年3月歌曲录音素材-储备店长班
 92  25年3月歌曲录音素材-门店端92
 76  25年8月歌曲录音素材-储备店长班
 68  25年5月歌曲录音素材-门店端68
 66  25年4月歌曲录音素材-储备店长班
 64  25年6月歌曲录音素材-储备店长班
 60  25年2月歌曲录音素材-门店端60
 52  25年7月歌曲录音素材-储备店长班
 46  25年9月歌曲录音素材-储备店长班
 46  25年8月歌曲录音素材-管培生
 45  25年1月歌曲录音素材-门店端45
 44  25年2月歌曲录音素材-储备店长班
 42  25年11月至12月歌曲录音素材-储备店长班
其余目录均 ≤26 条
```

### 2.4 数据侧

对 `train-data` 的本地统计结果如下：

- 总音频数：`2797`
- 总时长：约 `514.28` 小时
- 时长分布：
  - `P50` 约 `10.55` 分钟
  - `P95` 约 `14.49` 分钟
  - 最长约 `23.13` 分钟
- 采样率分布：
  - `48000 Hz`: `1474`
  - `44100 Hz`: `1249`
  - `16000 Hz`: `34`
  - `8000 Hz`: `23`
  - `24000 Hz`: `13`
  - `22050 Hz`: `4`
- 文件名标签粗分：
  - `自己/本人/自几/本` 类：`1250` 条，约 `229.33` 小时
  - `异性` 类：`757` 条，约 `139.15` 小时
  - `朋友/亲友` 类：`40` 条，约 `7.38` 小时
  - 未能直接从文件名判断的：`750` 条，约 `138.41` 小时
- 明显异常命名样本约 `105` 条，问题包括：
  - `.m4a..wav` / `.MP3..wav`
  - 全角、半角横线混用
  - `(1)` `(2)` 重复标记
  - `自几`、`本人-本` 等错别字或截断标签

从目录名看，较值得优先使用的高质量数据源是：

- `recordings_single_channel_denoised_cut80hz`：`680` 条，约 `126.27` 小时
- `歌曲录音素材单声道降噪去80hz以下版`：`410` 条，约 `75.79` 小时

这两部分合计大约 `202.06` 小时，适合作为第一轮扩训的核心干净子集。

## 3. 目标定义

本次训练目标不是重新训练一个新底模，而是：

- 从现有 `sovits_spk_1700h` 系列底模出发继续训练。
- 扩大模型对企业内部真实录歌场景、长时歌曲、人声录音条件差异的覆盖。
- 尽量保住当前模型已有的零样本音色迁移能力，避免因为脏数据一次性注入导致音色塌缩、杂音增加或说话人边界变差。

因此本计划采用“先干净、后扩大；先验证、后全量”的两阶段扩训。

## 4. 数据重构方案

### 4.1 先做统一命名和说话人归档

当前 `train-data` 不是训练代码直接可读的结构，建议整理成：

```text
data_samoye_expand_v1/
  raw_clean/<speaker_id>/*.wav
  waves-32k/<speaker_id>/*.wav
  waves-16k/<speaker_id>/*.wav
  specs/<speaker_id>/*.pt
  pitch/<speaker_id>/*.pit.npy
  hubert/<speaker_id>/*.vec.npy
  whisper/<speaker_id>/*.ppg.npy
  speaker/<speaker_id>/*.spk.npy
```

归档原则：

- `speaker_id` 以“人”为单位，而不是以月份、批次或歌曲为单位。
- 文件名里出现的工号前缀如 `ZY0021`、`JM0735` 可以保留在样本名里，但不要把它当成唯一 speaker id。
- 对同一人的 `自己`、`本人`、`异性`、`朋友` 录音，speaker id 应当一致，标签只作为样本属性保留。
- 对无法明确对应到人的 `unknown` 样本，先单独放入 `unknown_pool`，不直接混入第一轮训练。

### 4.2 先做一轮清洗

训练前至少做以下处理：

- 修正异常文件名，统一横线、空格、扩展名。
- 删除明显重复文件和同一音频的多版本残留。
- 剔除明显损坏文件和非单人主唱文件。
- 把 `8k/16k` 的低带宽音频先放入隔离集合，不进第一轮主训练。
- 对 `unknown`、`朋友/亲友`、`异性` 标签样本做人工抽检。

### 4.3 不建议直接拿整首歌做第一轮特征提取

虽然当前 loader 会在训练时随机裁片，但原始音频中位时长超过 10 分钟。直接对整首歌提全部特征会导致：

- 特征提取耗时极高。
- 磁盘占用过大。
- 长段无效伴奏、前奏、尾奏和静音被一起编码，拖低训练质量。
- speaker encoder 在线编码整段 wav，训练吞吐会明显下降。

建议先把每条长音频切成 `20` 到 `40` 秒的人声主导片段，再进入特征提取链路。切段规则建议：

- 优先保留主唱稳定段，尽量避开前奏、尾奏和长静音。
- 允许 `3` 到 `5` 秒重叠，降低切点损伤。
- 单段过短时丢弃，建议小于 `12` 秒不保留。
- 每个 speaker 每首歌最多保留 `2` 到 `4` 段，防止少数歌曲过度主导。

## 5. 分阶段扩训方案

### 阶段 A：`v1-core` 高质量子集验证

目的：

- 验证当前代码链路在企业内部数据上能稳定收敛。
- 尽量保住原底模能力，不让模型一开始就被脏数据拉偏。

建议纳入数据：

- `recordings_single_channel_denoised_cut80hz`
- `歌曲录音素材单声道降噪去80hz以下版`
- 其他目录中标签为 `自己/本人` 且采样率为 `44.1k/48k` 的样本

建议暂缓纳入：

- `8k/16k/22.05k/24k` 低采样率样本
- `unknown` 样本
- `朋友/亲友` 样本
- 明显命名异常但尚未人工确认的样本

规模目标：

- 第一轮控制在 `150` 到 `220` 小时之间即可，不追求一开始吃满全部 514 小时。

训练策略：

- 起点权重：以 `sovits_spk_1700h` 的现有 checkpoint 为起点。
- 训练方式：全参数解冻继续训练。
- **推荐加载方式**：使用 `pretrain` 模式（在 yaml 中将 `train.pretrain` 指向 1700h checkpoint），而不是 `--checkpoint_path`。原因：
  - `pretrain` 模式下 optimizer 从零初始化，学习率完全由 yaml 配置控制，不受原始训练 epoch 数影响。
  - `--checkpoint_path` 会恢复原始 optimizer 动量和 epoch 计数，如果原训练已跑数百 epoch，`ExponentialLR` 衰减后的实际学习率可能已极小（有效 lr = `lr_config × 0.999875^(restored_epoch-1)`），需要额外补偿。
  - 仓库现有的 finetune yaml（`sovits_spk_1700h_finetune_sing.yaml`）正是采用这种方式。
- 建议学习率：`2e-5` 到 `5e-5`（因为 optimizer 从零初始化，需要比 `--checkpoint_path` 模式稍高）
- 建议 epoch：`10` 到 `20`
- 建议命名：`sovits_spk_expand_v1_core`

理由：

- 当前代码没有分层冻结逻辑，最小改动方案就是直接全量续训。
- 用较小学习率从成熟底模继续训练，可以降低灾难性遗忘风险。
- `pretrain` 模式的学习率行为最可预测，避免了 `ExponentialLR` 历史衰减带来的不确定性。

### 阶段 B：`v2-expand` 扩大覆盖范围

在阶段 A 试听和验证通过后，继续加入：

- 非降噪但质量可接受的 `自己/本人` 样本
- 抽检通过的 `unknown` 样本
- 少量 `异性` 样本

不建议一次性把 `异性` 和 `unknown` 全加满。更稳的做法是：

- 先拼出 `train_core_plus_expand.txt`
- 保证新增脏数据不超过总训练量的 `30%` 到 `40%`
- 如果当前采样逻辑无法做权重采样，就通过“控制清单条数”来控制比例

训练策略：

- 从阶段 A 最优 checkpoint 继续。
- **加载方式**：同样推荐 `pretrain` 模式，将 yaml 的 `train.pretrain` 指向阶段 A 的最优 `.pt` 文件。
- 学习率降到 `1e-5` 到 `2e-5`
- 再训 `10` 到 `20` 个 epoch
- 命名建议：`sovits_spk_expand_v2_mix`

### 阶段 C：专项域适配

只有在业务上明确需要覆盖弱质录音时，再单独做第三阶段：

- 纳入 `8k/16k`、手机通话感较强、背景噪声大的样本
- 单独形成 `train_low_quality_adapt.txt`
- 训练 `3` 到 `5` 个 epoch
- 学习率控制在 `5e-6` 附近

这个阶段不要和主扩训阶段混在一起，否则很容易把主模型音质拉差。

## 6. 训练清单和验证集设计

### 6.1 不要再使用仓库里缺失的旧清单名

Q&A 已经说明，`files/train_filtered_1700_withoutnan.txt` 缺失时，应使用预处理脚本生成的新 `train.txt`。

建议本次直接生成并维护以下清单：

- `files/train_expand_v1_core.txt`
- `files/train_expand_v2_mix.txt`
- `files/train_expand_v3_low_quality.txt`
- `files/valid_expand.txt`

### 6.2 验证集要人工控质，不能只随机抽 10 条

当前 `preprocess_train.py` 默认只抽前 `10` 条作为验证，这对扩训不够。

建议验证集至少包含：

- `20` 到 `30` 个 speaker
- 每个 speaker `2` 到 `4` 条样本
- 同时覆盖：
  - 男/女
  - 门店端/储备店长班/职能/战区录音
  - 降噪和非降噪来源

另外建议准备一套独立人工试听集，用于对比：

- 原底模 checkpoint
- 阶段 A checkpoint
- 阶段 B checkpoint

## 7. 预处理落地步骤

以下流程尽量复用仓库现有脚本。

### 7.1 重采样

使用 `prepare/preprocess_a.py`：

```bash
python prepare/preprocess_a.py -w data_samoye_expand_v1/raw_clean -o data_samoye_expand_v1/waves-32k -s 32000 -t 0
python prepare/preprocess_a.py -w data_samoye_expand_v1/raw_clean -o data_samoye_expand_v1/waves-16k -s 16000 -t 0
```

说明：

- `waves-32k` 供 `spec` 和训练波形使用。
- `waves-16k` 供 `pitch / hubert / ppg / speaker` 使用。

### 7.2 特征提取

```bash
python prepare/preprocess_spec.py -w data_samoye_expand_v1/waves-32k -s data_samoye_expand_v1/specs -t 0
python prepare/preprocess_f0.py -w data_samoye_expand_v1/waves-16k -p data_samoye_expand_v1/pitch -t 0
python prepare/preprocess_hubert.py -w data_samoye_expand_v1/waves-16k -v data_samoye_expand_v1/hubert
python prepare/preprocess_ppg.py -w data_samoye_expand_v1/waves-16k -p data_samoye_expand_v1/whisper
python prepare/preprocess_speaker.py data_samoye_expand_v1/waves-16k data_samoye_expand_v1/speaker -t 0
python prepare/preprocess_train.py --src data_samoye_expand_v1
```

说明：

- 如果业务上更重视唱歌音高稳定，可把 `preprocess_f0.py` 换成 `preprocess_crepe.py` 做对比实验。
- `speaker` 虽然训练时不直接读取 `.spk.npy` 内容，但 `preprocess_train.py` 会校验它是否存在，所以仍应生成。

### 7.3 生成专用训练配置

建议新建一份配置，例如：

- `configs/sovits_spk_expand_v1_core.yaml`
- `configs/sovits_spk_expand_v2_mix.yaml`

以下是 `sovits_spk_expand_v1_core.yaml` 相对于基线配置 `sovits_spk_1700h.yaml` 的具体修改项：

```yaml
# === 需要修改的字段 ===
train:
  epochs: 20                                              # 从 1000 改为 20
  learning_rate: 3e-5                                     # 从 5e-5 适当降低
  pretrain: "./chkpt/sovits_spk_1700h/sovits_spk_1700h_0020.pt"  # 指向 1700h checkpoint

data:
  training_files: "files/train_expand_v1_core.txt"        # 新训练清单
  validation_files: "files/valid_expand.txt"              # 新验证清单

log:
  info_interval: 1                                        # 从 100 改为 1，方便逐 step 监控
  eval_interval: 1                                        # 从 2 改为 1，每 epoch 验证
  save_interval: 1                                        # 从 4 改为 1，每 epoch 保存
  keep_ckpts: 5                                           # 从 0 改为 5，自动清理旧 checkpoint
```

第一轮**绝对不要改**的字段（改动会导致 checkpoint 不兼容）：

- `sampling_rate: 32000`
- `hop_length: 320`
- `segment_size: 8000`
- `ppg_dim: 1280`
- `vec_dim: 256`
- `spk_dim: 256`
- `gin_channels: 256`
- `inter_channels: 192`
- `hidden_channels: 192`
- `filter_channels: 640`
- 所有 `gen` / `mpd` / `mrd` 下的结构参数

### 7.4 启动训练命令

```bash
# 阶段 A：v1-core 训练
python svc_trainer.py -c configs/sovits_spk_expand_v1_core.yaml -n sovits_spk_expand_v1_core

# Tensorboard 监控
tensorboard --logdir=logs/sovits_spk_expand_v1_core --port 12345
```

注意：这里**不需要**传 `--checkpoint_path` 参数，因为 yaml 中的 `train.pretrain` 已经指向了 1700h checkpoint。`svc_trainer.py` 会自动通过 pretrain 模式加载权重。

## 8. 推荐训练日程

### 第 1 周：数据治理

- 完成 speaker 归档规则
- 清洗异常命名
- 切段
- 产出 `v1-core`

### 第 2 周：特征提取和链路验收

- 跑通 `waves-32k / waves-16k / spec / pitch / hubert / ppg / speaker`
- 生成 `train_expand_v1_core.txt`
- 做一次 `100` 到 `300` step 的 smoke test，确认 loss、显存、I/O 没问题

### 第 3 周：阶段 A 训练

- 全参数续训 `10` 到 `20` epoch
- 每个 epoch 做试听对比
- 选出最优 checkpoint

### 第 4 周：阶段 B 扩大训练

- 加入第二批样本
- 继续训练 `10` 到 `20` epoch
- 比较是否出现杂音上升、音色变糊、辅音边界变差

## 9. 验收标准

### 9.1 Tensorboard 监控指标

训练日志每 step 输出格式为（`train.py:310-333`）：

```
epoch {epoch} | g {loss_g} m {mel_loss} s {stft_loss} d {loss_d} k {kl_f} r {kl_r} i {spk_loss} | step {step}
```

各项含义和正常范围：

| 指标 | 含义 | 正常趋势 | 异常信号 |
|------|------|---------|---------|
| `g` | 生成器总损失 | 稳定下降或平稳 | 持续上升或剧烈震荡 |
| `m` | Mel spectrogram L1 损失（权重 `c_mel=1.0`） | 稳定下降 | 突然上升说明音频质量问题 |
| `s` | 多分辨率 STFT 损失（权重 `c_stft=9`，是最大权重项） | 稳定下降 | 持续不降说明频谱重建有问题 |
| `d` | 判别器损失 | 在 `1.0` 到 `4.0` 间波动 | 趋近 0（判别器过强）或远大于 5（判别器过弱） |
| `k` | 前向 flow KL 散度（权重 `c_kl=0.2`） | 缓慢下降 | 急剧上升提示分布偏移 |
| `r` | 反向 flow KL 散度（权重 `×0.5`） | 缓慢下降 | 同上 |
| `i` | 说话人 cosine embedding 损失（权重 `×2`） | 保持稳定或缓慢下降 | **持续上升是关键警报**，说明新数据的说话人分布与底模冲突 |

### 9.2 主观评估

建议至少同时看以下四类指标：

- 训练稳定性：
  - 上述 7 项损失是否符合正常趋势
- 主观音质：
  - 是否出现明显毛刺、齿音放大、混响拖尾异常
- 音色保持：
  - 目标 speaker 的听感一致性是否比原底模更强
- 泛化能力：
  - 对未参与训练的新 speaker reference，是否仍能维持可用的零样本迁移效果

### 9.3 阶段间决策规则

- 如果阶段 A 的 `spk_loss` 在 5 epoch 内持续上升，**立即停训**，检查数据中是否混入了非人声或标签错误的样本。
- 如果阶段 B 让 `mel_loss` 和 `stft_loss` 比阶段 A 最优点上升超过 `20%`，应优先保留阶段 A 的 checkpoint 作为主线模型。
- 如果阶段 B 的主观试听出现明显音质下降，但客观损失仍在下降，大概率是新增脏数据比例过高，应减少脏数据占比后重新训练。

## 10. 最终建议

如果目标是”用现有 `train-data` 给 SaMoye 底模解冻扩大训练”，最推荐的执行顺序是：

1. 只拿两类高质量降噪数据（`recordings_single_channel_denoised_cut80hz` + `歌曲录音素材单声道降噪去80hz以下版`，合计约 1090 条、202 小时）加 `自己/本人` 样本做第一轮扩训。
2. 从现有 `1700h` 模型 checkpoint 通过 `pretrain` 模式加载权重，optimizer 从零初始化，全参数小学习率（`2e-5` 到 `5e-5`）续训，不改模型结构。
3. 把 `unknown`、`异性`、低采样率样本放到第二轮或第三轮，绝不直接混训。
4. **必须在训练前把长音频切成 20-40 秒中短段**——这不只是为了减少特征提取开销，更是因为 speaker encoder 在训练时会在线加载整段 wav 计算 embedding，10 分钟整歌会严重拖慢训练吞吐。
5. 用新的 `train_expand_*.txt` 和 `valid_expand.txt` 管理数据版本，不再依赖仓库里缺失的老清单。
6. 重点监控 `spk_loss`（说话人损失）——这是扩训能否保住零样本音色迁移能力的关键指标。

这条路线对当前仓库侵入最小，成功率最高，也最符合 SaMoye 现有代码的实际工作方式。
