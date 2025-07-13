import os,sys,torch,warnings,pdb
warnings.filterwarnings("ignore")
import librosa
import importlib
import  numpy as np
import hashlib , math
from tqdm import tqdm
from uvr5_pack.lib_v5 import spec_utils
from uvr5_pack.utils import _get_name_params,inference
from uvr5_pack.lib_v5.model_param_init import ModelParameters
from scipy.io import wavfile

import sys
current_path = (os.path.dirname(os.path.abspath(__file__)))

class  _audio_pre_():
    def __init__(self, model_path,device,is_half):
        self.model_path = model_path
        self.device = device
        self.data = {
            # Processing Options
            'postprocess': False,
            'tta': False,
            # Constants
            'window_size': 512,
            'agg': 10,
            'high_end_process': 'mirroring',
        }
        nn_arch_sizes = [
            31191, # default
            33966,61968, 123821, 123812, 537238 # custom
        ]
        self.nn_architecture = list('{}KB'.format(s) for s in nn_arch_sizes)
        model_size = math.ceil(os.stat(model_path ).st_size / 1024)
        nn_architecture = '{}KB'.format(min(nn_arch_sizes, key=lambda x:abs(x-model_size)))
        nets = importlib.import_module('uvr5_pack.lib_v5.nets' + f'_{nn_architecture}'.replace('_{}KB'.format(nn_arch_sizes[0]), ''), package=None)
        model_hash = hashlib.md5(open(model_path,'rb').read()).hexdigest()
        param_name ,model_params_d = _get_name_params(model_path , model_hash)

        mp = ModelParameters(os.path.join(current_path, model_params_d))
        model = nets.CascadedASPPNet(mp.param['bins'] * 2)
        cpk = torch.load( model_path , map_location='cpu')  
        model.load_state_dict(cpk)
        model.eval()
        if(is_half==True):model = model.half().to(device)
        else:model = model.to(device)

        self.mp = mp
        self.model = model

    def _path_audio_(self, music_file ,ins_path,vocal_path):
        X_wave, y_wave, X_spec_s, y_spec_s = {}, {}, {}, {}
        bands_n = len(self.mp.param['band'])
        # print(bands_n)
        for d in range(bands_n, 0, -1): 
            bp = self.mp.param['band'][d]
            if d == bands_n: # high-end band
                # X_wave[d], _ = librosa.core.load(
                #     music_file, False,  sr=bp['sr'],dtype=np.float32, res_type=bp['res_type'])
                X_wave[d], _ = librosa.load(
                    music_file,sr=bp["sr"])
                if X_wave[d].ndim == 1:
                    X_wave[d] = np.asfortranarray([X_wave[d], X_wave[d]])
            else: # lower bands
                X_wave[d] = librosa.resample(X_wave[d+1], orig_sr=self.mp.param['band'][d+1]['sr'], target_sr=bp['sr'], res_type=bp['res_type'])
            # Stft of wave source
            X_spec_s[d] = spec_utils.wave_to_spectrogram_mt(X_wave[d], bp['hl'], bp['n_fft'], self.mp.param['mid_side'], self.mp.param['mid_side_b2'], self.mp.param['reverse'])
            # pdb.set_trace()
            if d == bands_n and self.data['high_end_process'] != 'none':
                input_high_end_h = (bp['n_fft']//2 - bp['crop_stop']) + ( self.mp.param['pre_filter_stop'] - self.mp.param['pre_filter_start'])
                input_high_end = X_spec_s[d][:, bp['n_fft']//2-input_high_end_h:bp['n_fft']//2, :]

        X_spec_m = spec_utils.combine_spectrograms(X_spec_s, self.mp)
        aggresive_set = float(self.data['agg']/100)
        aggressiveness = {'value': aggresive_set, 'split_bin': self.mp.param['band'][1]['crop_stop']}
        with torch.no_grad():
            pred, X_mag, X_phase = inference(X_spec_m,self.device,self.model, aggressiveness,self.data)
        # Postprocess
        if self.data['postprocess']:
            pred_inv = np.clip(X_mag - pred, 0, np.inf)
            pred = spec_utils.mask_silence(pred, pred_inv)
        y_spec_m = pred * X_phase
        v_spec_m = X_spec_m - y_spec_m

        if self.data['high_end_process'].startswith('mirroring'):
            input_high_end_ = spec_utils.mirroring(self.data['high_end_process'], y_spec_m, input_high_end, self.mp)
            wav_instrument = spec_utils.cmb_spectrogram_to_wave(y_spec_m, self.mp,input_high_end_h, input_high_end_)
        else:
            wav_instrument = spec_utils.cmb_spectrogram_to_wave(y_spec_m, self.mp)
        wavfile.write(ins_path, self.mp.param['sr'], (np.array(wav_instrument)*32768).astype("int16"))  #
        
        if self.data['high_end_process'].startswith('mirroring'):
            input_high_end_ = spec_utils.mirroring(self.data['high_end_process'],  v_spec_m, input_high_end, self.mp)
            wav_vocals = spec_utils.cmb_spectrogram_to_wave(v_spec_m, self.mp, input_high_end_h, input_high_end_)
        else:
            wav_vocals = spec_utils.cmb_spectrogram_to_wave(v_spec_m, self.mp)
        wavfile.write(vocal_path, self.mp.param['sr'], (np.array(wav_vocals)*32768).astype("int16"))

if __name__ == '__main__':
    device = 'cuda'
    is_half=True
    model_path='/home/pengfei/projects/audio_enhancement_mq/tmp/UVR5_Linux/models/VR_Models/9_HP2-UVR.pth'
    pre_fun = _audio_pre_(model_path=model_path,device=device,is_half=True)
    audio_path = 'audio.aac'
    save_path = 'opt'
    pre_fun._path_audio_(audio_path , save_path,save_path)
