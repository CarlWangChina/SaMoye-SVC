import sys
sys.path.append('.') 
import time
import os
import random
import traceback
import numpy as np
import torch
import torch.utils.data
import librosa
import ffmpeg
import torch.nn.functional as F
from tqdm import tqdm
from module.mel_processing import spectrogram_torch
from feature_extractor import cnhubert, rmvpe_pitch
from torch.utils.data import DataLoader
from functools import cache

# Hyperparameters: Dataset
MIN_DATA_NUM = 100

# Hyperparameters: Audio
WAV_MAX_VALUE = 32768.0
SAMPLE_RATE = 32000
FILTER_LENGTH = 2048
HOP_LENGTH = 640
WIN_LENGTH = 2048
SAMPLING_RATE = 32000

MIN_DURATION = 0.6
MAX_DURATION = 54

# Seed
random.seed(3507)



'''
Supportive Functions
'''
@cache
def load_audio(file, sr, load_duration=5.0):
    try:
        # 防止小白拷路径头尾带了空格和"和回车
        file = (
            file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        out, _ = (
            ffmpeg.input(file, ss=random.uniform(0, max(0, librosa.get_duration(path=file) - load_duration)), t=load_duration)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")
    return np.frombuffer(out, np.float32).flatten()

def get_nearest_even_number(n):
    return 2 * ((n // 2) + 1)

'''
Data Loader for Training
'''
class AudioTextLoader(torch.utils.data.Dataset):
    def __init__(self, wav_roots, eval_mode=False):
        self.eval_mode = eval_mode
        self.max_wav_value = WAV_MAX_VALUE
        self.sampling_rate = SAMPLE_RATE
        self.filter_length = FILTER_LENGTH
        self.hop_length = HOP_LENGTH
        self.win_length = WIN_LENGTH
        # SSL extractor CNHubert and F0 predictor RMVPE
        self.cnhubert_model = cnhubert.get_cnhubert_model('pretrained_models/chinese-hubert-base')
        self.rmvpe = rmvpe_pitch.get_f0_predictor('pretrained_models/rmvpe.pt')

        for p in self.cnhubert_model.parameters():
            p.requires_grad = False

        audio_files = []
        assert type(wav_roots) == list or type(wav_roots) == str
        if type(wav_roots) == str:
            wav_roots = [wav_roots]

        for wav_root in wav_roots:
            for path, _, files in os.walk(wav_root):
                for file in files:
                    if file.endswith('.wav') or file.endswith('.mp3'):
                        audio_files.append(os.path.join(path, file))
        
        datas = list(set(audio_files))

        cur_fine_datas = datas
        cur_data_num = len(cur_fine_datas)

        # if amount of data less than MIN_DATA_NUM, repeat the data till it reaches MIN_DATA_NUM
        if (cur_data_num < MIN_DATA_NUM):
            datas = []
            for _ in range(max(2, int(MIN_DATA_NUM / cur_data_num))):
                datas += cur_fine_datas
        # shuffle the datas
        random.shuffle(datas)

        final_datas = []
        lengths = []

        for audiopath in tqdm(datas):
            size = os.path.getsize(audiopath)
            duration = size / self.sampling_rate / 2

            if not duration:
                print(f"Skip Blank Audio {audiopath}...")
                continue

            if self.eval_mode or MIN_DURATION < duration < MAX_DURATION:
                final_datas.append(audiopath)
                lengths.append(size // (2 * self.hop_length))
            else:
                continue

        assert len(final_datas) > 1

        self.datas = final_datas
        self.lengths = lengths

    def get_f0_from_wav(self, wav, f0_predictor):
        f0 = f0_predictor.infer_from_audio(wav)
        return f0

    def get_hubert_features(self, wav, cnhubert_model):
        # max_tmp = np.abs(wav).max()
        # maxx = 0.95
        # alpha = 0.5
        # tmp_audio32 = (wav / max_tmp * (maxx * alpha * 32768)) + ((1 - alpha) * 32768) * wav

        tensor_wav16 = torch.from_numpy(wav)
        ssl = cnhubert_model.model(tensor_wav16.unsqueeze(0))["last_hidden_state"].transpose(1, 2).cpu()
        return ssl
 
    def get_all_features_from_wav(self, wav_file, sr, f0_predictor, cnhubert_model):
        wav = load_audio(wav_file, sr)

        f0 = self.get_f0_from_wav(wav, f0_predictor)

        spec = spectrogram_torch(
            torch.FloatTensor(wav).unsqueeze(0), FILTER_LENGTH,
            sr, HOP_LENGTH, WIN_LENGTH, center=False
        )
        spec = torch.squeeze(spec, 0)

        wav_16k = librosa.resample(wav, orig_sr=32000, target_sr=16000)
        # s_time = time.time()
        ssl = self.get_hubert_features(wav_16k, cnhubert_model).detach()
        # print("SSL Time: ", time.time() - s_time)
    
        if (ssl.shape[-1] != spec.shape[-1]):
            ssl_type = ssl.dtype
            ssl = F.pad(ssl.float(), (0, 1), mode="replicate").to(ssl_type)
        
        f0 = torch.FloatTensor(f0)
        wav = torch.FloatTensor(wav).unsqueeze(0)
        return spec, wav, ssl, f0

    def __getitem__(self, index):
        return self.get_all_features(self.datas[index])
    
    def get_all_features(self, wav_pth):
        try:
            spec, wav, ssl, f0 = self.get_all_features_from_wav(wav_pth, self.sampling_rate, self.rmvpe, self.cnhubert_model)
        except:
            traceback.print_exc()
            spec = torch.zeros(1025, 100)
            wav = torch.zeros(1, 100 * self.hop_length)
            ssl = torch.zeros(1, 768, 100)
            f0 = torch.zeros(100)
            print("Shit Audio Path: ", wav_pth)

        return (ssl, spec, wav, f0)



    def __len__(self):
        return len(self.datas)


class AudioTextCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        ssl_batch, spec_batch, wav_batch, f0_batch = zip(*batch)

        _, ndx_spec_decrease = torch.sort(
            torch.LongTensor([spec.size(1) for spec in spec_batch]),
            dim=0, descending=True
        )

        # ssl: pad to nearest even number of max lenght
        max_ssl_len = max([ssl.size(2) for ssl in ssl_batch])
        max_ssl_len = get_nearest_even_number(max_ssl_len)
        # spec: pad to nearest even number of max length
        max_spec_len = max([spec.size(1) for spec in spec_batch])
        max_spec_len = get_nearest_even_number(max_spec_len)
        # wav/f0: pad to max length
        max_wav_len = max([wavs.size(1) for wavs in wav_batch])

        ssl_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        f0_length = torch.LongTensor(len(batch))

        ssl_padded = torch.FloatTensor(len(batch), ssl_batch[0].size(1), max_ssl_len)
        spec_padded = torch.FloatTensor(len(batch), spec_batch[0].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        f0_pad = torch.FloatTensor(len(batch), max_ssl_len)

        spec_padded.zero_()
        wav_padded.zero_()
        ssl_padded.zero_()
        f0_pad.zero_()

        for i in range(len(ndx_spec_decrease)):
            ssl = ssl_batch[ndx_spec_decrease[i]]
            ssl_padded[i, :, :ssl.size(2)] = ssl[0, :, :]
            ssl_lengths[i] = ssl.size(2)

            spec = spec_batch[ndx_spec_decrease[i]]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = wav_batch[ndx_spec_decrease[i]]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            f0 = f0_batch[ndx_spec_decrease[i]]
            f0_pad[i, :f0.size(0)] = f0
            f0_length[i] = f0.size(0)

        return ssl_padded, ssl_lengths, spec_padded, spec_lengths,\
                    wav_padded, wav_lengths, f0_pad, f0_length

class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        i = len(buckets) - 1
        while i >= 0:
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)
            i -= 1

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            rem = num_samples_bucket - len_bucket
            ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]

            ids_bucket = ids_bucket[self.rank::self.num_replicas]

            for j in range(len(ids_bucket) // self.batch_size):
                batch = [bucket[idx] for idx in ids_bucket[j * self.batch_size:(j + 1) * self.batch_size]]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size