import torch

from modules.nsf_hifigan.env import AttrDict
from modules.nsf_hifigan.models import Generator


# noinspection SpellCheckingInspection
class NSFHiFiGANONNX(torch.nn.Module):
    def __init__(self, attrs: dict):
        super().__init__()
        self.generator = Generator(AttrDict(attrs))

    def forward(self, mel: torch.Tensor, f0: torch.Tensor):
        mel = mel.transpose(1, 2) * 2.30259
        wav = self.generator(mel, f0)
        return wav.squeeze(1)
