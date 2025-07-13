# Copyright (c) 2024 MusicBeing Project. All Rights Reserved.
#
# Author: Feee <cgoxopx@outlook.com>
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from denoiser import pretrained
from denoiser.dsp import convert_audio
from auenhan.fbdenoiser import *
import torchaudio

# 加载音频文件
input_audio = '/home/pengfei/projects/audio_enhancement_mq/tests/data/huanjie_nvzhong_sing.wav'
output_audio = '/home/pengfei/projects/audio_enhancement_mq/tests/outputs/huanjie_nvzhong_sing.wav'

# 使用示例：
# 创建一个 Denoiser 实例
denoiser = Denoiser()

# 对文件进行降噪处理
denoiser.denoise_file(input_audio, output_audio)
print("denoise completed")

# # 对音频 tensor 进行降噪处理
# audio_tensor, sample_rate = torchaudio.load(input_audio)
# enhanced_tensor, enhanced_rate = denoiser.denoise_tensor(audio_tensor, sample_rate)
# print(enhanced_tensor.shape, enhanced_rate)