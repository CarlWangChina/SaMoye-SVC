from collections import OrderedDict

import tqdm
import json
import pathlib

import numpy as np
import torch
from typing import Dict

from basics.base_svs_infer import BaseSVSInfer
from modules.fastspeech.param_adaptor import VARIANCE_CHECKLIST
from modules.fastspeech.tts_modules import LengthRegulator
from modules.toplevel import DiffSingerAcoustic, ShallowDiffusionOutput
from modules.vocoders.registry import VOCODERS
from utils import load_ckpt
from utils.hparams import hparams
from utils.infer_utils import cross_fade, resample_align_curve, save_wav
from utils.phoneme_utils import build_phoneme_list
from utils.text_encoder import TokenTextEncoder
import time

class DiffSingerAcousticInfer(BaseSVSInfer):
    
    def __init__(self, device=None, load_model=True, load_vocoder=True, ckpt_steps=None):
        super().__init__(device=device)
        if load_model:
            self.variance_checklist = []

            self.variances_to_embed = set()

            if hparams.get('use_energy_embed', False):
                self.variances_to_embed.add('energy')
            if hparams.get('use_breathiness_embed', False):
                self.variances_to_embed.add('breathiness')

            self.ph_encoder = TokenTextEncoder(vocab_list=build_phoneme_list())
            if hparams['use_spk_id']:
                with open(pathlib.Path(hparams['work_dir']) / 'spk_map.json', 'r', encoding='utf8') as f:
                    self.spk_map = json.load(f)
                assert isinstance(self.spk_map, dict) and len(self.spk_map) > 0, 'Invalid or empty speaker map!'
                assert len(self.spk_map) == len(set(self.spk_map.values())), 'Duplicate speaker id in speaker map!'
            self.model = self.build_model(ckpt_steps=ckpt_steps)
            self.lr = LengthRegulator().to(self.device)
        if load_vocoder:
            self.vocoder = self.build_vocoder()
        self.total_time = 0
        self.times = []
        self.batch_size = hparams['batch_size']

    def build_model(self, ckpt_steps=None):
        model = DiffSingerAcoustic(
            vocab_size=len(self.ph_encoder),
            out_dims=hparams['audio_num_mel_bins']
        ).eval().to(self.device)
        load_ckpt(model, hparams['work_dir'], ckpt_steps=ckpt_steps,
                  prefix_in_ckpt='model', strict=True, device=self.device)
        return model

    def build_vocoder(self):
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
        mel2ph = self.lr(durations, txt_tokens == 0)  # => [B=1, T]
        batch['mel2ph'] = mel2ph
        length = mel2ph.size(1)  # => T

        summary['tokens'] = txt_tokens.size(1)
        summary['frames'] = length
        summary['seconds'] = '%.2f' % (length * self.timestep)

        if hparams['use_spk_id']:
            spk_mix_id, spk_mix_value = self.load_speaker_mix(
                param_src=param, summary_dst=summary, mix_mode='frame', mix_length=length
            )
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
                try:
                    # print(f'Embedding variance: {v_name}, {param[v_name]}, {param[f"{v_name}_timestep"]} {self.timestep} {length}')
                    batch[v_name] = torch.from_numpy(resample_align_curve(
                        np.array(param[v_name].split(), np.float32),
                        original_timestep=float(param[f'{v_name}_timestep']),
                        target_timestep=self.timestep,
                        align_length=length
                    )).to(self.device)[None]
                    summary[v_name] = 'manual'
                except KeyError:
                    # set to one
                    batch[v_name] = torch.ones(1, length, dtype=torch.float32).to(self.device)
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

        # print(f'[{idx}]\t' + ', '.join(f'{k}: {v}' for k, v in summary.items()))
        self.times.append(float(summary['seconds']))
        self.total_time += float(summary['seconds'])
        return batch

    def create_tensor_array(self, sample, key, tensor_type):
        max_length = max(entry[key].shape[1] for entry in sample)
        tensor_array = torch.zeros((len(sample), max_length), dtype=tensor_type).to(self.device)
        for i, entry in enumerate(sample):
            tensor_array[i, :entry[key].shape[1]] = entry[key]
        return tensor_array

    @torch.no_grad()
    def forward_model(self, sample):
        txt_tokens = self.create_tensor_array(sample, 'tokens', torch.long)
        mel2ph_tensor = self.create_tensor_array(sample, 'mel2ph', torch.long)
        f0_tensor = self.create_tensor_array(sample, 'f0', torch.float32)
        key_shift_tensor = self.create_tensor_array(sample, 'key_shift', torch.float32)
        speed_tensor = self.create_tensor_array(sample, 'speed', torch.float32)
        # print(f"tokens: {txt_tokens.shape} mel2ph: {mel2ph_tensor.shape} f0: {f0_tensor.shape} key_shift: {key_shift_tensor.shape}")
        variances = {
            v_name: self.create_tensor_array(sample, v_name, torch.float32)
            for v_name in self.variances_to_embed
        }
        if hparams['use_spk_id']:
            spk_mix_id = torch.zeros((len(sample), 1, 1), dtype=torch.int64).to(self.device)
            for i, entry in enumerate(sample):
                spk_mix_id[i][0][0] = entry['spk_mix_id']
            spk_mix_value = torch.zeros((len(sample), 1, 1), dtype=torch.float32).to(self.device)
            for i, entry in enumerate(sample):
                spk_mix_value[i][0][0] = entry['spk_mix_value']
            spk_embed = self.model.fs2.spk_embed(spk_mix_id)
            spk_mix_value = spk_mix_value.unsqueeze(3)
            # perform mixing on spk embed
            spk_mix_embed = torch.sum(
                spk_embed * spk_mix_value,  # => [B, T, N, H]
                dim=2, keepdim=False
            )  # => [B, T, H]
            # print(f"spk_mix_id: {spk_mix_id.shape} spk_mix_value: {spk_mix_value.shape} spk_embed: {spk_embed.shape} spk_mix_embed: {spk_mix_embed.shape}")
            
            # spk_mix_id = sample[0]['spk_mix_id']
            # spk_mix_value = sample[0]['spk_mix_value']
            # # perform mixing on spk embed
            # spk_mix_embed = torch.sum(
            #     self.model.fs2.spk_embed(spk_mix_id) * spk_mix_value.unsqueeze(3),  # => [B, T, N, H]
            #     dim=2, keepdim=False
            # )  # => [B, T, H]
        else:
            spk_mix_embed = None
        
        
        mel_pred: ShallowDiffusionOutput = self.model(
            txt_tokens, mel2ph=mel2ph_tensor, f0=f0_tensor, **variances,
            key_shift=key_shift_tensor, speed=speed_tensor,
            spk_mix_embed=spk_mix_embed, infer=True
        )
        return mel_pred.diff_out

    @torch.no_grad()
    def run_vocoder(self, spec, **kwargs):
        y = self.vocoder.spec2wav_torch(spec, **kwargs)
        return y[None]

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
            if (i+1) % self.batch_size == 0 or i == len(params) - 1:
                batches.append(small_batch)
                # print(f'batch: {i} , {len(small_batch)}')
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
                # print(f"Shape of output: {mel_pred_out.shape}")
                if save_mel:
                    for z in range(len(mel_pred_out[0])):
                        mel_pred = mel_pred_out[z]
                        param = params[j * self.batch_size + z]
                        result.append({
                            'offset': param.get('offset', 0.),
                            'mel': mel_pred.cpu(),
                            'f0': batch[z]['f0'].cpu()
                        })
                else:
                    for z in range(mel_pred_out.shape[0]):
                        param = params[j * self.batch_size + z]
                        len_wav = batch[z]['f0'].shape[1]
                        # print(f"len_wav: {len_wav} {mel_pred_out[z][:len_wav].unsqueeze(0).shape} {batch[z]['f0'].shape}")
                        waveform_pred = self.run_vocoder(mel_pred_out[z][:len_wav].unsqueeze(0), f0=batch[z]['f0'])[0].cpu().numpy()
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
