import copy
import json

import tqdm
import pathlib
from collections import OrderedDict

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import interpolate
from typing import List, Tuple

from basics.base_svs_infer import BaseSVSInfer
from modules.fastspeech.tts_modules import (
    LengthRegulator, RhythmRegulator,
    mel2ph_to_dur
)
from modules.fastspeech.param_adaptor import VARIANCE_CHECKLIST
from modules.toplevel import DiffSingerVariance
from utils import load_ckpt
from utils.hparams import hparams
from utils.infer_utils import resample_align_curve
from utils.phoneme_utils import build_phoneme_list
from utils.pitch_utils import interp_f0
from utils.text_encoder import TokenTextEncoder
import time
from einops import rearrange
from onnx_pitch_run import DsPitch
from onnx_duration_run import DsDuration

class DiffSingerONNXVarianceInfer(BaseSVSInfer):
    def __init__(
            self, device=None, ckpt_steps=None,
            predictions: set = None
    ):
        super().__init__(device=device)
        self.ph_encoder = TokenTextEncoder(vocab_list=build_phoneme_list())

        self.dsPitch = DsPitch()
        self.dsDuration = DsDuration()

        self.lr = LengthRegulator()
        self.rr = RhythmRegulator()
        smooth_kernel_size = round(hparams['midi_smooth_width'] / self.timestep)
        self.smooth = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=smooth_kernel_size,
            bias=False,
            padding='same',
            padding_mode='replicate'
        ).eval().to(self.device)
        smooth_kernel = torch.sin(torch.from_numpy(
            np.linspace(0, 1, smooth_kernel_size).astype(np.float32) * np.pi
        ).to(self.device))
        smooth_kernel /= smooth_kernel.sum()
        self.smooth.weight.data = smooth_kernel[None, None]

        glide_types = hparams.get('glide_types', [])
        assert 'none' not in glide_types, 'Type name \'none\' is reserved and should not appear in glide_types.'
        self.glide_map = {
            'none': 0,
            **{
                typename: idx + 1
                for idx, typename in enumerate(glide_types)
            }
        }

        self.auto_completion_mode = len(predictions) == 0
        self.global_predict_dur = 'dur' in predictions and hparams['predict_dur']
        self.global_predict_pitch = 'pitch' in predictions and hparams['predict_pitch']
        self.variance_prediction_set = predictions.intersection(VARIANCE_CHECKLIST)
        self.global_predict_variances = len(self.variance_prediction_set) > 0

        self.total_time = 0
        self.times = []

        self.batch_size = hparams['batch_size']

    @torch.no_grad()
    def preprocess_input(
            self, param, idx=0,
            load_dur: bool = False,
            load_pitch: bool = False
    ):
        """
        :param param: one segment in the .ds file
        :param idx: index of the segment
        :param load_dur: whether ph_dur is loaded
        :param load_pitch: whether pitch is loaded
        :return: batch of the model inputs
        """
        batch = {}
        summary = OrderedDict()
        txt_tokens = torch.LongTensor([self.ph_encoder.encode(param['ph_seq'].split())]).to(self.device)  # [B=1, T_ph]
        T_ph = txt_tokens.shape[1]
        batch['tokens'] = txt_tokens
        
        ph_num = torch.from_numpy(np.array([param['ph_num'].split()], np.int64)).to(self.device)  # [B=1, T_w]
        batch['ph_num'] = ph_num
        ph2word = self.lr(ph_num)  # => [B=1, T_ph]
        T_w = int(ph2word.max())
        batch['ph2word'] = ph2word

        # old code
        # note_seq = torch.FloatTensor(
        #     [(librosa.note_to_midi(n, round_midi=False) if n != 'rest' else -1) for n in param['note_seq'].split()]
        # ).to(self.device)[None]  # [B=1, T_n]
        
        # T_n = note_seq.shape[1]
        # note_dur_sec = torch.from_numpy(np.array([param['note_dur'].split()], np.float32)).to(self.device)  # [B=1, T_n]
        # note_acc = torch.round(torch.cumsum(note_dur_sec, dim=1) / self.timestep + 0.5).long()
        # note_dur = torch.diff(note_acc, dim=1, prepend=note_acc.new_zeros(1, 1))
        # mel2note = self.lr(note_dur)  # [B=1, T_s]
        # T_s = mel2note.shape[1]
        # old code end

        # new code
        note_midi = np.array(
            [(librosa.note_to_midi(n, round_midi=False) if n != 'rest' else -1) for n in param['note_seq'].split()],
            dtype=np.float32
        )
        note_rest = note_midi < 0
        if np.all(note_rest):
            # All rests, fill with constants
            note_midi = np.full_like(note_midi, fill_value=60.)
        else:
            # Interpolate rest values
            interp_func = interpolate.interp1d(
                np.where(~note_rest)[0], note_midi[~note_rest],
                kind='nearest', fill_value='extrapolate'
            )
            note_midi[note_rest] = interp_func(np.where(note_rest)[0])
        note_midi = torch.from_numpy(note_midi).to(self.device)[None]  # [B=1, T_n]
        note_rest = torch.from_numpy(note_rest).to(self.device)[None]  # [B=1, T_n]
        
        T_n = note_midi.shape[1]
        note_dur_sec = torch.from_numpy(np.array([param['note_dur'].split()], np.float32)).to(self.device)  # [B=1, T_n]
        note_acc = torch.round(torch.cumsum(note_dur_sec, dim=1) / self.timestep + 0.5).long()
        note_dur = torch.diff(note_acc, dim=1, prepend=note_acc.new_zeros(1, 1))
        mel2note = self.lr(note_dur)  # [B=1, T_s]
        T_s = mel2note.shape[1]
        # new code end

        summary['words'] = T_w
        summary['notes'] = T_n
        summary['tokens'] = T_ph
        summary['frames'] = T_s
        summary['seconds'] = '%.2f' % (T_s * self.timestep)

        if hparams['use_spk_id']:
            ph_spk_mix_id, ph_spk_mix_value = self.load_speaker_mix(
                param_src=param, summary_dst=summary, mix_mode='token', mix_length=T_ph
            )
            spk_mix_id, spk_mix_value = self.load_speaker_mix(
                param_src=param, summary_dst=summary, mix_mode='frame', mix_length=T_s
            )
            batch['ph_spk_mix_id'] = ph_spk_mix_id
            batch['ph_spk_mix_value'] = ph_spk_mix_value
            batch['spk_mix_id'] = spk_mix_id
            batch['spk_mix_value'] = spk_mix_value

        if load_dur:
            # Get mel2ph if ph_dur is needed
            ph_dur_sec = torch.from_numpy(
                np.array([param['ph_dur'].split()], np.float32)
            ).to(self.device)  # [B=1, T_ph]
            ph_acc = torch.round(torch.cumsum(ph_dur_sec, dim=1) / self.timestep + 0.5).long()
            ph_dur = torch.diff(ph_acc, dim=1, prepend=ph_acc.new_zeros(1, 1))
            mel2ph = self.lr(ph_dur, txt_tokens == 0)
            if mel2ph.shape[1] != T_s:  # Align phones with notes
                mel2ph = F.pad(mel2ph, [0, T_s - mel2ph.shape[1]], value=mel2ph[0, -1])
                ph_dur = mel2ph_to_dur(mel2ph, T_ph)
            # Get word_dur from ph_dur and ph_num
            word_dur = note_dur.new_zeros(1, T_w + 1).scatter_add(
                1, ph2word, ph_dur
            )[:, 1:]  # => [B=1, T_w]
        else:
            ph_dur = None
            mel2ph = None
            # Get word_dur from note_dur and note_slur
            is_slur = torch.BoolTensor([[int(s) for s in param['note_slur'].split()]]).to(self.device)  # [B=1, T_n]
            note2word = torch.cumsum(~is_slur, dim=1)  # [B=1, T_n]
            word_dur = note_dur.new_zeros(1, T_w + 1).scatter_add(
                1, note2word, note_dur
            )[:, 1:]  # => [B=1, T_w]

        batch['ph_dur'] = ph_dur
        batch['mel2ph'] = mel2ph

        mel2word = self.lr(word_dur)  # [B=1, T_s]
        if mel2word.shape[1] != T_s:  # Align words with notes
            mel2word = F.pad(mel2word, [0, T_s - mel2word.shape[1]], value=mel2word[0, -1])
            word_dur = mel2ph_to_dur(mel2word, T_w)
        batch['word_dur'] = word_dur
        
        # old code
        # batch['note_midi'] = note_seq
        # batch['note_dur'] = note_dur
        # batch['note_rest'] = note_seq < 0
        # old code end
        # new code
        batch['note_midi'] = note_midi
        batch['note_dur'] = note_dur
        batch['note_rest'] = note_rest
        # new code end
        if hparams.get('use_glide_embed', False) and param.get('note_glide') is not None:
            batch['note_glide'] = torch.LongTensor(
                [[self.glide_map.get(x, 0) for x in param['note_glide'].split()]]
            ).to(self.device)
        else:
            batch['note_glide'] = torch.zeros(1, T_n, dtype=torch.long, device=self.device)
        batch['mel2note'] = mel2note

        # Calculate frame-level MIDI pitch, which is a step function curve
        # old code
        # frame_midi_pitch = torch.gather(
        #     F.pad(note_seq, [1, 0]), 1, mel2note
        # )  # => frame-level MIDI pitch, [B=1, T_s]
        # rest = (frame_midi_pitch < 0)[0].cpu().numpy()
        # frame_midi_pitch = frame_midi_pitch[0].cpu().numpy()
        # interp_func = interpolate.interp1d(
        #     np.where(~rest)[0], frame_midi_pitch[~rest],
        #     kind='nearest', fill_value='extrapolate'
        # )
        # frame_midi_pitch[rest] = interp_func(np.where(rest)[0])
        # frame_midi_pitch = torch.from_numpy(frame_midi_pitch[None]).to(self.device)
        # base_pitch = self.smooth(frame_midi_pitch)
        # batch['base_pitch'] = base_pitch
        # new code
        # Calculate and smoothen the frame-level MIDI pitch, which is a step function curve
        frame_midi_pitch = torch.gather(
            F.pad(note_midi, [1, 0]), 1, mel2note
        )  # => frame-level MIDI pitch, [B=1, T_s]
        base_pitch = self.smooth(frame_midi_pitch)
        batch['base_pitch'] = base_pitch

        if ph_dur is not None:
            # Phone durations are available, calculate phoneme-level MIDI.
            mel2pdur = torch.gather(F.pad(ph_dur, [1, 0], value=1), 1, mel2ph)  # frame-level phone duration
            ph_midi = frame_midi_pitch.new_zeros(1, T_ph + 1).scatter_add(
                1, mel2ph, frame_midi_pitch / mel2pdur
            )[:, 1:]
        else:
            # Phone durations are not available, calculate word-level MIDI instead.
            mel2wdur = torch.gather(F.pad(word_dur, [1, 0], value=1), 1, mel2word)
            w_midi = frame_midi_pitch.new_zeros(1, T_w + 1).scatter_add(
                1, mel2word, frame_midi_pitch / mel2wdur
            )[:, 1:]
            # Convert word-level MIDI to phoneme-level MIDI
            ph_midi = torch.gather(F.pad(w_midi, [1, 0]), 1, ph2word)
        ph_midi = ph_midi.round().long()
        batch['midi'] = ph_midi

        if load_pitch:
            f0 = resample_align_curve(
                np.array(param['f0_seq'].split(), np.float32),
                original_timestep=float(param['f0_timestep']),
                target_timestep=self.timestep,
                align_length=T_s
            )
            batch['pitch'] = torch.from_numpy(
                librosa.hz_to_midi(interp_f0(f0)[0]).astype(np.float32)
            ).to(self.device)[None]

        # if self.model.predict_dur:
        if True:    
            if load_dur:
                summary['ph_dur'] = 'manual'
            elif self.auto_completion_mode or self.global_predict_dur:
                summary['ph_dur'] = 'auto'
            else:
                summary['ph_dur'] = 'ignored'

        # if self.model.predict_pitch:
        if True:
            if load_pitch:
                summary['pitch'] = 'manual'
            elif self.auto_completion_mode or self.global_predict_pitch:
                summary['pitch'] = 'auto'

                # Load expressiveness
                expr = param.get('expr', 1.)
                if isinstance(expr, (int, float, bool)):
                    summary['expr'] = f'static({expr:.3f})'
                    batch['expr'] = torch.FloatTensor([expr]).to(self.device)[:, None]  # [B=1, T=1]
                else:
                    summary['expr'] = 'dynamic'
                    expr = resample_align_curve(
                        np.array(expr.split(), np.float32),
                        original_timestep=float(param['expr_timestep']),
                        target_timestep=self.timestep,
                        align_length=T_s
                    )
                    batch['expr'] = torch.from_numpy(expr.astype(np.float32)).to(self.device)[None]

            else:
                summary['pitch'] = 'ignored'

        # if self.model.predict_variances:
        #     for v_name in self.model.variance_prediction_list:
        #         if self.auto_completion_mode and param.get(v_name) is None or v_name in self.variance_prediction_set:
        #             summary[v_name] = 'auto'
        #         else:
        #             summary[v_name] = 'ignored'

        print(f'[{idx}]\t' + ', '.join(f'{k}: {v}' for k, v in summary.items()))
        self.times.append(float(summary['seconds']))
        self.total_time += float(summary['seconds'])
        
        return batch

    def run_inference(
            self, params,
            out_dir: pathlib.Path = None,
            title: str = None,
            num_runs: int = 1,
            seed: int = -1
    ):
        batches = []
        small_batch = []
        predictor_flags: List[Tuple[bool, bool, bool]] = []
        
        for i, param in enumerate(params):
            param: dict
            flag = (self.global_predict_dur, self.global_predict_pitch, False) # predict_dur, predict_pitch, predict_variances
            predictor_flags.append(flag)
            batch = self.preprocess_input(
                param, idx=i,
                load_dur=not flag[0] and (flag[1] or flag[2]),
                load_pitch=not flag[1] and flag[2]
            )
            small_batch.append(batch)
            if (i+1) % self.batch_size == 0 or i == len(params) - 1:
                batches.append(small_batch)
                # print(f'batch: {i} , {len(small_batch)}')
                small_batch = []

        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(num_runs):
            results = []
            for j, batch in enumerate(tqdm.tqdm(batches, 
            desc='infer segments', total=len(batches)
            )):
                start_time = time.time()
                param = params[j*self.batch_size]
                if 'seed' in param:
                    torch.manual_seed(param["seed"] & 0xffff_ffff)
                    torch.cuda.manual_seed_all(param["seed"] & 0xffff_ffff)
                elif seed >= 0:
                    torch.manual_seed(seed & 0xffff_ffff)
                    torch.cuda.manual_seed_all(seed & 0xffff_ffff)

                dur_pred_flag = True
                if dur_pred_flag == True:
                    # 使用这里的batch 进行推理
                    dur_pred_out = self.dsDuration.Process(batch)
                    # for out in dur_pred_out:
                    #     print(f"Shape of output: {out.shape}")

                    if dur_pred_out is not None and (self.auto_completion_mode or self.global_predict_dur):
                        for z in range(len(dur_pred_out[0])):
                            n_frames = batch[z]['tokens'].shape[1]
                            dur_pred = dur_pred_out[0][z][:n_frames]
                            # batch[z]['ph_dur'] = torch.from_numpy(dur_pred).to(dtype=torch.float32, device=self.device)
                            # batch[z]['ph_dur'] = rearrange(batch[z]['ph_dur'], 'h -> 1 h')
                            ph_dur_sec = torch.from_numpy(
                                    np.array(dur_pred, np.float32)
                                ).to(self.device)  # [B=1, T_ph]
                            ph_dur_sec = rearrange(ph_dur_sec, 'h -> 1 h')
                            # print(f'ph_dur_sec: {ph_dur_sec}')
                            ph_acc = torch.round(torch.cumsum(ph_dur_sec, dim=1)+0.5).long()
                            batch[z]['ph_dur'] = torch.diff(ph_acc, dim=1, prepend=ph_acc.new_zeros(1, 1))
                            mel2ph = self.lr(batch[z]['ph_dur'], batch[z]['tokens'] == 0)
                            T_s = batch[z]['mel2note'].shape[1]
                            T_ph = batch[z]['tokens'].shape[1]
                            if mel2ph.shape[1] != T_s:  # Align phones with notes
                                mel2ph = F.pad(mel2ph, [0, T_s - mel2ph.shape[1]], value=mel2ph[0, -1])
                                batch[z]['ph_dur'] = mel2ph_to_dur(mel2ph, T_ph)
                        # print(f': {batch}')
                
                pitch_pred_out = self.dsPitch.Process(batch)
                # for out in pitch_pred_out:
                #     print(f"Shape of output: {out.shape}")
                
                for z in range(len(pitch_pred_out[0])):
                    param, flag = params[j*self.batch_size + z], predictor_flags[j*self.batch_size + z]
                    param_copy = copy.deepcopy(param)
                    
                    if dur_pred_out is not None and (self.auto_completion_mode or self.global_predict_dur):
                        n_frames = batch[z]['tokens'].shape[1]
                        dur_pred = dur_pred_out[0][z][:n_frames]
                        # print(f'dur_pred: {dur_pred}')
                        param_copy['ph_dur'] = ' '.join(str(round(dur, 6)) for dur in (dur_pred * self.timestep).tolist())
                    
                    if pitch_pred_out is not None and (self.auto_completion_mode or self.global_predict_pitch):
                        pitch = pitch_pred_out[0][z]
                        n_frames = batch[z]['base_pitch'].shape[1]
                        pitch_pred = np.array(pitch[:n_frames]).flatten()
                        f0_pred = librosa.midi_to_hz(pitch_pred)
                        # f0_pred 低于50置为0
                        f0_pred[f0_pred < 50] = 0
                        # print(f'f0_pred: {f0_pred.shape} pitch_pred: {pitch_pred.shape}')
                        param_copy['f0_seq'] = ' '.join([str(round(freq, 1)) for freq in f0_pred.tolist()])
                        param_copy['f0_timestep'] = str(self.timestep)

                    # Restore ph_spk_mix and spk_mix
                    if 'ph_spk_mix' in param_copy and 'spk_mix' in param_copy:
                        if 'ph_spk_mix_backup' in param_copy:
                            if param_copy['ph_spk_mix_backup'] is None:
                                del param_copy['ph_spk_mix']
                            else:
                                param_copy['ph_spk_mix'] = param_copy['ph_spk_mix_backup']
                            del param['ph_spk_mix_backup']
                        if 'spk_mix_backup' in param_copy:
                            if param_copy['ph_spk_mix_backup'] is None:
                                del param_copy['spk_mix']
                            else:
                                param_copy['spk_mix'] = param_copy['spk_mix_backup']
                            del param['spk_mix_backup']

                    results.append(param_copy)

            if num_runs > 1:
                filename = f'{title}-{str(i).zfill(3)}.ds'
            else:
                filename = f'{title}.ds'
            save_path = out_dir / filename
            with open(save_path, 'w', encoding='utf8') as f:
                print(f'| save params: {save_path}')
                json.dump(results, f, ensure_ascii=False, indent=2)

    