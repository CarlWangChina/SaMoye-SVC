import torch
import torchaudio
from typing import Tuple
from denoiser import pretrained
from denoiser.dsp import convert_audio

class Denoiser:
    def __init__(self, model_name: str = 'dns64', device: str = 'cuda'):
        """
        初始化 Denoiser 类并加载预训练模型。

        参数:
            model_name (str): 模型名称 ('dns64', 'dns48', 等). 默认 'dns64'.
            device (str): 设备名称 ('cpu', 'cuda', 等). 默认 'cuda'.
        """
        self.device = device
        # 选择模型
        if model_name == 'dns64':
            self.model = pretrained.dns64().to(device)
        elif model_name == 'dns48':
            self.model = pretrained.dns48().to(device)
        elif model_name == 'master64':
            self.model = pretrained.master64().to(device)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        
        # 设置模型为评估模式
        self.model.eval()
        print(self.model.sample_rate)

    @torch.inference_mode()
    def denoise_file(self, input_path: str, output_path: str) -> None:
        """
        对文件进行降噪处理。

        参数:
            input_path (str): 输入音频文件路径。
            output_path (str): 输出音频文件路径。
        """
        assert isinstance(input_path, str), "input_path 必须是字符串类型"
        assert isinstance(output_path, str), "output_path 必须是字符串类型"
        
        # 加载音频文件
        wav, sr = torchaudio.load(input_path)
        
        # 将音频转换为模型需要的格式
        wav = convert_audio(wav, sr, self.model.sample_rate, self.model.chin)
        
        # 执行降噪
        with torch.no_grad():
            enhanced = self.model(wav.unsqueeze(0).to(self.device))[0]
        
        # 保存结果
        torchaudio.save(output_path, enhanced.cpu(), self.model.sample_rate)
    
    @torch.inference_mode()
    def denoise_tensor(self, audio_tensor: torch.Tensor, sample_rate: int) -> Tuple[torch.Tensor, int]:
        """
        对音频 tensor 进行降噪处理。

        参数:
            audio_tensor (torch.Tensor): 输入音频 tensor (2D).
            sample_rate (int): 输入音频采样率。
        
        返回:
            Tuple[torch.Tensor, int]: 降噪后的音频 tensor 和其采样率。
        """
        assert isinstance(audio_tensor, torch.Tensor), "audio_tensor 必须是 torch.Tensor 类型"
        assert audio_tensor.dim() == 2, "audio_tensor 必须是二维 (channels, samples)"
        assert isinstance(sample_rate, int), "sample_rate 必须是整型"
        
        # 将音频转换为模型需要的格式
        wav = convert_audio(audio_tensor, sample_rate, self.model.sample_rate, self.model.chin)
        
        # 执行降噪
        with torch.no_grad():
            enhanced = self.model(wav.unsqueeze(0).to(self.device))[0]
        
        return enhanced.cpu(), self.model.sample_rate

