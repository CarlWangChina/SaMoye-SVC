import numpy as np
import os
from pathlib import Path
from utils.hparams import hparams
import torch

class DSONNXSpeakerEmbedManager():
    def __init__(self, ds_config, root_path):
        self.ds_config = ds_config
        self.root_path = root_path
        self.speaker_embeds = None
        self.speaker_id = {}
        self.device = None
    def load_speaker_embed(self, speaker):
        path = self.root_path / f"{speaker}.emb"
        if path.exists():
            with open(path, 'rb') as file:
                # 从二进制文件中读取单精度浮点数
                data = torch.from_numpy(np.fromfile(file, dtype=np.float32, count=self.ds_config.get('hiddenSize'))).to(self.device)
            return data
        else:
            raise Exception(f"Speaker embed file {path} not found")
    
    def get_speaker_embeds(self):
        if self.speaker_embeds is None:
            if self.ds_config['speakers'] is None:
                return None
            else:
                embeds = torch.zeros((len(self.ds_config['speakers']), self.ds_config.get('hiddenSize')), dtype=torch.float32).to(self.device)
                speaker_id = {}
                for spk_id in range(len(self.ds_config['speakers'])):
                    embeds[spk_id] = self.load_speaker_embed(self.ds_config['speakers'][spk_id])
                    speaker_id[self.ds_config['speakers'][spk_id]] = spk_id
                    # print(f"Loaded speaker embed for {self.ds_config['speakers'][spk_id]}")
                self.speaker_embeds = embeds
                self.speaker_id = speaker_id
        return self.speaker_embeds


    def get_speaker_index_by_name(self, speaker_name):
        speaker_index = self.speaker_id.get(speaker_name)
        return speaker_index

    def get_speaker_embeds_by_spk_mix_id(self, spk_mix_id):
        speaker_embeds = self.get_speaker_embeds()
        # print(speaker_embeds.shape)
        spk_embed_result = torch.zeros((len(spk_mix_id), 1, self.ds_config.get('hiddenSize')), dtype=torch.float32).to(self.device)
        for i in range(len(spk_mix_id)):
            spk_embed_result[i][0] = speaker_embeds[spk_mix_id[i]]
        return spk_embed_result

    def phrase_speaker_embed_by_frame(self, spk_mix_id, spk_mix_value):
        self.device = spk_mix_id.device
        speaker_embeds = self.get_speaker_embeds_by_spk_mix_id(spk_mix_id)
        # print(f"speaker_embeds.shape: {speaker_embeds.shape}")
        hidden_size = self.ds_config['hiddenSize']

        spk_embed_result = speaker_embeds * spk_mix_value
        spk_embed_tensor = spk_embed_result.cpu().numpy().astype(np.float32) # .reshape((len(spk_mix_value), 1, hidden_size))
        # print(f"spk_embed_tensor.shape: {spk_embed_tensor.shape}")
        return spk_embed_tensor