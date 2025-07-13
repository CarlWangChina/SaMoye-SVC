import copy
import json
import yaml

import tqdm
import pathlib
from collections import OrderedDict

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import interpolate
from typing import List, Tuple, Dict
import time

from basics.base_svs_infer import BaseSVSInfer
from modules.fastspeech.param_adaptor import VARIANCE_CHECKLIST
from modules.fastspeech.tts_modules import LengthRegulator
from modules.toplevel import ShallowDiffusionOutput
from modules.vocoders.registry import VOCODERS
from utils.hparams import hparams
from utils.infer_utils import cross_fade, resample_align_curve, save_wav
from utils.phoneme_utils import build_phoneme_list
from utils.text_encoder import TokenTextEncoder

from onnx_speaker_run import DSONNXSpeakerEmbedManager
from onnx_vocoder_run import DsVocoder
import onnxruntime

class DiffSingerONNXAcousticInfer(BaseSVSInfer):
    def __init__(self, device=None, load_model=True, load_vocoder=True):
        super().__init__(device=device)

        self.dsConfig = self.load_dsconifg_yaml(hparams['onnx_model_dir'] /  "dsconfig.yaml")
        self.dsConfig['hiddenSize'] = 256
        print(f"| dsConfig: {self.dsConfig}")
        self.phonemes = self.load_phonemes(hparams['onnx_model_dir'] / self.dsConfig['phonemes'])
        print(f"| phonemes: {self.phonemes}")
        if load_model:
            self.variance_checklist = []
            self.variances_to_embed = set()

            if hparams.get('use_energy_embed', False):
                self.variances_to_embed.add('energy')
            if hparams.get('use_breathiness_embed', False):
                self.variances_to_embed.add('breathiness')

            self.ph_encoder = TokenTextEncoder(vocab_list=self.phonemes)
            if hparams['use_spk_id']:
                self.speakerEmbedManager = DSONNXSpeakerEmbedManager(self.dsConfig, hparams['onnx_model_dir'])
                
            self.model = self.build_model()
            self.lr = LengthRegulator().to(self.device)
        if load_vocoder:
            self.vocoder = self.build_vocoder()
        self.total_time = 0
        self.times = []
        self.batch_size = hparams['batch_size']
        
    def load_phonemes(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            phonemes_list = [line.strip() for line in file.readlines() if line.strip() != '<PAD>']

        return phonemes_list

    def load_dsconifg_yaml(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            params = yaml.safe_load(f)
        return params

    def build_model(self):
        linguisticModel_path = hparams['onnx_model_dir'] / self.dsConfig['acoustic']
        acousticModel = onnxruntime.InferenceSession(linguisticModel_path, providers=['CUDAExecutionProvider'])
        # print(f"acousticModel: {acousticModel.get_providers()}")
        return acousticModel

    def build_vocoder(self):
        if (hparams['onnx_model_dir'] / "dsvocoder" / "vocoder.yaml").exists():
            # 读取自带的vocoder
            vocoder = DsVocoder(hparams['onnx_model_dir'] / "dsvocoder")
        else:
            if hparams['vocoder'] in VOCODERS:
                vocoder = VOCODERS[hparams['vocoder']]()
            else:
                vocoder = VOCODERS[hparams['vocoder'].split('.')[-1]]()
            vocoder.to_device(self.device)
        return vocoder

    def preprocess_input(self, param, idx=0):
        """
        :param param: one segment in the .ds file
        :param idx: index of the segment
        :return: batch of the model inputs
        """
        batch = {}
        summary = OrderedDict()
        txt_tokens = torch.LongTensor([self.ph_encoder.encode(param['ph_seq'])]).to(self.device)  # => [B, T_txt]
        batch['tokens'] = txt_tokens

        ph_dur = torch.from_numpy(np.array(param['ph_dur'].split(), np.float32)).to(self.device)
        ph_acc = torch.round(torch.cumsum(ph_dur, dim=0) / self.timestep + 0.5).long()
        durations = torch.diff(ph_acc, dim=0, prepend=torch.LongTensor([0]).to(self.device))[None]  # => [B=1, T_txt]
        batch['durations'] = durations
        mel2ph = self.lr(durations, txt_tokens == 0)  # => [B=1, T]
        batch['mel2ph'] = mel2ph
        length = mel2ph.size(1)  # => T

        summary['tokens'] = txt_tokens.size(1)
        summary['frames'] = length
        summary['seconds'] = '%.2f' % (length * self.timestep)

        if hparams['use_spk_id']:
            spk_mix_id = torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device)
            spk_mix_id[0][0][0] = 0
            spk_mix_value = torch.zeros((1, 1, 1), dtype=torch.float32).to(self.device)
            spk_mix_value[0][0][0] = 1
            batch['spk_mix_id'] = spk_mix_id
            batch['spk_mix_value'] = spk_mix_value
            

        batch['f0'] = torch.from_numpy(resample_align_curve(
            np.array(param['f0_seq'].split(), np.float32),
            original_timestep=float(param['f0_timestep']),
            target_timestep=self.timestep,
            align_length=length
        )).to(self.device)[None]

        for v_name in VARIANCE_CHECKLIST:
            if v_name in self.variances_to_embed:
                batch[v_name] = torch.from_numpy(resample_align_curve(
                    np.array(param[v_name].split(), np.float32),
                    original_timestep=float(param[f'{v_name}_timestep']),
                    target_timestep=self.timestep,
                    align_length=length
                )).to(self.device)[None]
                summary[v_name] = 'manual'

        if hparams.get('use_key_shift_embed', False):
            shift_min, shift_max = hparams['augmentation_args']['random_pitch_shifting']['range']
            gender = param.get('gender')
            if gender is None:
                gender = 0.
            if isinstance(gender, (int, float, bool)):  # static gender value
                summary['gender'] = f'static({gender:.3f})'
                key_shift_value = gender * shift_max if gender >= 0 else gender * abs(shift_min)
                batch['key_shift'] = torch.FloatTensor([key_shift_value]).to(self.device)[:, None]  # => [B=1, T=1]
            else:
                summary['gender'] = 'dynamic'
                gender_seq = resample_align_curve(
                    np.array(gender.split(), np.float32),
                    original_timestep=float(param['gender_timestep']),
                    target_timestep=self.timestep,
                    align_length=length
                )
                gender_mask = gender_seq >= 0
                key_shift_seq = gender_seq * (gender_mask * shift_max + (1 - gender_mask) * abs(shift_min))
                batch['key_shift'] = torch.clip(
                    torch.from_numpy(key_shift_seq.astype(np.float32)).to(self.device)[None],  # => [B=1, T]
                    min=shift_min, max=shift_max
                )

        if hparams.get('use_speed_embed', False):
            if param.get('velocity') is None:
                summary['velocity'] = 'default'
                batch['speed'] = torch.FloatTensor([1.]).to(self.device)[:, None]  # => [B=1, T=1]
            else:
                summary['velocity'] = 'manual'
                speed_min, speed_max = hparams['augmentation_args']['random_time_stretching']['range']
                speed_seq = resample_align_curve(
                    np.array(param['velocity'].split(), np.float32),
                    original_timestep=float(param['velocity_timestep']),
                    target_timestep=self.timestep,
                    align_length=length
                )
                batch['speed'] = torch.clip(
                    torch.from_numpy(speed_seq.astype(np.float32)).to(self.device)[None],  # => [B=1, T]
                    min=speed_min, max=speed_max
                )

        print(f'[{idx}]\t' + ', '.join(f'{k}: {v}' for k, v in summary.items()))

        return batch

    def create_tensor_array(self, sample, key, tensor_type):
        max_length = max(entry[key].shape[1] for entry in sample)
        tensor_array = torch.zeros((len(sample), max_length), dtype=tensor_type).to(self.device)
        for i, entry in enumerate(sample):
            tensor_array[i, :entry[key].shape[1]] = entry[key]
        return tensor_array

    def process_entries(self, sample, entry_key, dtype):
        max_length = max(entry[entry_key].shape[1] for entry in sample)

        # 创建空的 NumPy 数组，用于存放整合后的数据
        entry_array = np.zeros((len(sample), max_length), dtype=dtype)

        # 将数据填充到 NumPy 数组中
        for i, entry in enumerate(sample):
            entry_array[i, :entry[entry_key].shape[1]] = entry[entry_key].cpu().numpy()

        return entry_array

    @torch.no_grad()
    def forward_model(self, sample):
        # Acoustic ONNX model infer
        # 1.获取模型的所有输入和输出节点名称
        # 获取模型的输入信息
        # input_details = self.model.get_inputs()
        # for input_detail in input_details:
        #     print(f"Input Name: {input_detail.name}, Input Shape: {input_detail.shape}")
        # Input Name: tokens, Input Shape: [1, 'n_tokens']
        # Input Name: durations, Input Shape: [1, 'n_tokens']
        # Input Name: f0, Input Shape: [1, 'n_frames']
        # Input Name: spk_embed, Input Shape: [1, 'n_frames', 256]
        # Input Name: depth, Input Shape: []
        # Input Name: speedup, Input Shape: []
        # 2.获取模型的输出信息
        # output_details = self.model.get_outputs()
        # for output_detail in output_details:
        #     print(f"Output Name: {output_detail.name}, Output Shape: {output_detail.shape}")
        # Output Name: mel, Output Shape: [1, 'n_frames', 128]

        # mel_pred: ShallowDiffusionOutput = self.model(
        #     txt_tokens, mel2ph=mel2ph_tensor, f0=f0_tensor, **variances,
        #     key_shift=key_shift_tensor, speed=speed_tensor,
        #     spk_mix_embed=spk_mix_embed, infer=True
        # )

        txt_tokens_array = self.process_entries(sample, 'tokens', np.int64)
        durations_array = self.process_entries(sample, 'durations', np.int64)
        f0_array = self.process_entries(sample, 'f0', np.float32)
        # print(f"tokens: {txt_tokens.shape} mel2ph: {mel2ph_tensor.shape} f0: {f0_tensor.shape} key_shift: {key_shift_tensor.shape}")
        variances = {
            v_name: self.process_entries(sample, v_name, np.float32)
            for v_name in self.variances_to_embed
        }
        # breathiness_array = np.full_like(f0_array, 100).astype(np.float32)
        # energy_array = np.full_like(f0_array, 0).astype(np.float32)
        # velocity_array = np.full_like(f0_array, 100).astype(np.float32)
        # gender_array = np.full_like(f0_array, 0).astype(np.float32)
        depth = [self.dsConfig['max_depth']] if 'max_depth' in self.dsConfig else [hparams['K_step_infer']]
        speedup = [hparams['pndm_speedup']]
        if hparams['use_spk_id']:     
            spk_mix_id = torch.zeros((len(sample), 1, 1), dtype=torch.int64).to(self.device)
            for i, entry in enumerate(sample):
                spk_mix_id[i][0][0] = entry['spk_mix_id']
            spk_mix_value = torch.zeros((len(sample), 1, 1), dtype=torch.float32).to(self.device)
            for i, entry in enumerate(sample):
                spk_mix_value[i][0][0] = entry['spk_mix_value']
            spk_embed_array = self.speakerEmbedManager.phrase_speaker_embed_by_frame(spk_mix_id, spk_mix_value)
        
        acousticInputs = {
            'tokens': txt_tokens_array,
            'durations': durations_array,
            'f0': f0_array,
            'spk_embed': spk_embed_array,
            'depth': depth,
            'speedup': speedup
        }
        # for name, tensor in acousticInputs.items():
        #     try:
        #         print(f"Shape of {name}: {tensor.shape}, content: {tensor}")
        #     except:
        #         print(f"Shape of {name}: {tensor}")
        mel_pred = self.model.run(None, acousticInputs)[0]
        mel_pred_np = np.array([np.asarray(array) for array in mel_pred])
        return mel_pred_np

    @torch.no_grad()
    def run_vocoder(self, mel, f0):
        # 1.获取模型的所有输入和输出节点名称
        # 获取模型的输入信息
        # input_details = self.vocoder.session.get_inputs()
        # for input_detail in input_details:
        #     print(f"Input Name: {input_detail.name}, Input Shape: {input_detail.shape}")
        # Input Name: mel, Input Shape: [1, 'n_frames', 128]
        # Input Name: f0, Input Shape: [1, 'n_frames']
        f0_array = f0.cpu().numpy()
        vocoderInputs = {
            'mel': mel,
            'f0': f0_array
        }
        waveform_pred = self.vocoder.session.run(None, vocoderInputs)
        waveform_pred_np = np.array([np.asarray(array) for array in waveform_pred])
        return waveform_pred_np[None]


    def run_inference(
            self, params,
            out_dir: pathlib.Path = None,
            title: str = None,
            num_runs: int = 1,
            spk_mix: Dict[str, float] = None,
            seed: int = -1,
            save_mel: bool = False
    ):
        batches = []
        small_batch = []
        for i, param in enumerate(params):
            batch = self.preprocess_input(param, idx=i)
            small_batch.append(batch)
            if (i + 1) % self.batch_size == 0 or i == len(params) - 1:
                batches.append(small_batch)
                print(f'batch: {i} , {len(small_batch)}')
                small_batch = []

        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = '.wav' if not save_mel else '.mel.pt'
        for i in range(num_runs):
            if save_mel:
                result = []
            else:
                result = np.zeros(0)
            current_length = 0

            
            for j, batch in enumerate(
                tqdm.tqdm(batches, desc='infer segments', total=len(batches))
                ):
                start_time = time.time()
                param = params[j * self.batch_size]
                if 'seed' in param:
                    torch.manual_seed(param["seed"] & 0xffff_ffff)
                    torch.cuda.manual_seed_all(param["seed"] & 0xffff_ffff)
                elif seed >= 0:
                    torch.manual_seed(seed & 0xffff_ffff)
                    torch.cuda.manual_seed_all(seed & 0xffff_ffff)

                mel_pred_out = self.forward_model(batch)
                # mel_pred_out = np.squeeze(mel_pred_out, axis=(0))

                # print(f"mel_pred_out: {mel_pred_out.shape}")
                # if save_mel:
                #     for z in range(len(mel_pred_out[0])):
                #         mel_pred = mel_pred_out[z]
                #         param = params[j * self.batch_size + z]
                #         result.append({
                #             'offset': param.get('offset', 0.),
                #             'mel': mel_pred.cpu(),
                #             'f0': batch[z]['f0'].cpu()
                #         })
                
                for z in range(mel_pred_out.shape[0]):
                    param = params[j * self.batch_size + z]
                    len_wav = batch[z]['f0'].shape[1]
                    mel_pred_squeeze = mel_pred_out[z, np.newaxis, :len_wav, :]
                    # print(f"len_wav: {len_wav} {mel_pred_squeeze.shape} {batch[z]['f0'].shape}")
                    waveform_pred = self.run_vocoder(mel_pred_squeeze, f0=batch[z]['f0'])[0][0][0]
                    print(f"waveform_pred: {waveform_pred.shape}")
                    silent_length = round(param.get('offset', 0) * hparams['audio_sample_rate']) - current_length
                    if silent_length >= 0:
                        result = np.append(result, np.zeros(silent_length))
                        result = np.append(result, waveform_pred)
                    else:
                        result = cross_fade(result, waveform_pred, current_length + silent_length)
                    current_length = current_length + silent_length + waveform_pred.shape[0]

            if num_runs > 1:
                filename = f'{title}-{str(i).zfill(3)}{suffix}'
            else:
                filename = title + suffix
            save_path = out_dir / filename
            if save_mel:
                print(f'| save mel: {save_path}')
                torch.save(result, save_path)
            else:
                print(f'| save audio: {save_path}')
                save_wav(result, save_path, hparams['audio_sample_rate'])
