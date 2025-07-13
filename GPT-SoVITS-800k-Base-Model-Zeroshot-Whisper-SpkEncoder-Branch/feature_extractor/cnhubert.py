import torch
from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)

import torch.nn as nn


class CNHubert(nn.Module):
    def __init__(self, cnhubert_base_path):
        super().__init__()
        self.model = HubertModel.from_pretrained(cnhubert_base_path)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            cnhubert_base_path
        )

    def forward(self, x):
        input_values = self.feature_extractor(
            x, return_tensors="pt", sampling_rate=16000
        ).input_values.to(x.device)
        feats = self.model(input_values)["last_hidden_state"]
        return feats

def get_cnhubert_model(cnhubert_base_path):
    model = CNHubert(cnhubert_base_path)
    model.eval()
    return model

def get_content(hubert_model, wav_16k_tensor):
    with torch.no_grad():
        feats = hubert_model(wav_16k_tensor)
    return feats.transpose(1, 2)


if __name__ == "__main__":
    model = get_cnhubert_model('pretrained_models/chinese-hubert-base')
    src_path = "dummy_datasets/hutao.wav_0000000000_0000171200.wav"
    print(model.feature_extractor(src_path))
