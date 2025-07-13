import sys
import os
from omegaconf import OmegaConf
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from vits.models import SynthesizerTrn
from vits_decoder.discriminator import Discriminator


def main():
    hp = OmegaConf.load("configs/zhangjian_read_20s.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_g = SynthesizerTrn(
        hp.data.filter_length // 2 + 1,
        hp.data.segment_size // hp.data.hop_length,
        hp
    ).to(device)
    model_d = Discriminator(hp).to(device)

    totalParams = {}
    ans = 0
    # for name, layer in model_g.named_parameters():
    for name, layer in model_d.named_parameters():
        partName = name.split(".")[0]
        if not partName in totalParams:
            totalParams[partName] = 0

        layerParam = layer.numel()
        ans += layerParam
        print(f"{name} has params -> {layerParam}")
        totalParams[partName] += layerParam

    print(f"Total params -> {totalParams}")
    print(f"Total params -> {sum(totalParams.values())}")


if __name__ == "__main__":
    main()
