import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"uvr"))
import torch
import torchaudio
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
import demucs.pretrained
import demucs.audio
import demucs.apply
import uvr.separate
from auenhan.fbdenoiser import Denoiser

class DemuxExtractor:
    def __init__(
            self,
            demucs_models: str = "htdemucs" ,
            device: str = "cuda",
            use_denoise: bool = True
    ):
        self.demucs_models = demucs.pretrained.get_model(demucs_models).to(device)
        self.demucs_sampling_rate = self.demucs_models.samplerate
        self.demucs_num_channels = self.demucs_models.audio_channels
        self.device = device
        
        self.denoiser = Denoiser(device=device) if use_denoise else None

    # Constants
    DEMUCS_DRUMS_INDEX = 0
    DEMUCS_BASS_INDEX = 1
    DEMUCS_OTHER_INDEX = 2
    DEMUCS_VOCAL_INDEX = 3

    @torch.inference_mode()
    def process_data_batch(self, batch: dict):
        assert batch is not None and not batch["audio_data"] is None and batch["audio_data"].dim() == 2
        waveform = batch["audio_data"]
        sample_rate = batch["sampling_rate"]
        
        waveform_resampled = demucs.audio.convert_audio(
            waveform, sample_rate, self.demucs_sampling_rate, self.demucs_num_channels
        ).reshape(1, 2, -1)

        stems = demucs.apply.apply_model(
            self.demucs_models, waveform_resampled.to(self.device)
        ).to("cpu")[0]
        
        acc = (stems[self.DEMUCS_BASS_INDEX] + stems[self.DEMUCS_OTHER_INDEX] + stems[self.DEMUCS_DRUMS_INDEX])/3
        vocal = stems[self.DEMUCS_VOCAL_INDEX]

        vocal_sr = self.demucs_sampling_rate
        if self.denoiser is not None:
            vocal, vocal_sr = self.denoiser.denoise_tensor(vocal, vocal_sr)
        
        os.makedirs(os.path.dirname(batch["vocal_path"]), exist_ok=True)
        os.makedirs(os.path.dirname(batch["acc_path"]), exist_ok=True)

        torchaudio.save(
            batch["vocal_path"],
            vocal,
            vocal_sr,
        )
        torchaudio.save(
            batch["acc_path"],
            acc,
            self.demucs_sampling_rate,
        )
    
    def process_file(self,infile,outfile_acc,outfile_vocal):
        audio, sampling_rate = torchaudio.load(infile)
        batch = {
                "file_path": infile,
                "vocal_path": outfile_vocal,
                "acc_path": outfile_acc,
                "audio_data": audio,
                "sampling_rate": sampling_rate
            }
        self.process_data_batch(batch)

class UVRExtractor:
    def __init__(self,model: str, device: str="cuda"):
        self.model = uvr.separate._audio_pre_(model, device, True)
    @torch.inference_mode()
    def process_file(self,infile,outfile_acc,outfile_vocal):
        # self.model._path_audio_(audio_path , save_path,save_path)
        self.model._path_audio_(infile, outfile_acc, outfile_vocal)
