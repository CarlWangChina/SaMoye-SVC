import os
from onnxruntime import InferenceSession
from pathlib import Path
import yaml
from utils.hparams import hparams

class DsVocoder:
    def __init__(self, name):
        try:
            self.location = name
            config_path = self.location / "vocoder.yaml"
            with open(config_path, 'r', encoding='utf-8') as config_file:
                self.config = yaml.safe_load(config_file)
            model_path = self.location / self.config['model']
        except Exception as ex:
            raise Exception(f"| Error loading vocoder {name}. Please download vocoder from https://github.com/xunmengshe/OpenUtau/wiki/Vocoders")

        self.session = InferenceSession(model_path, providers=['CUDAExecutionProvider'])
        print(f"| Loaded vocoder {name}, {self.session.get_providers()}")
    def frame_ms(self):
        return 1000.0 * self.config['hop_size'] / self.config['sample_rate']
    
    