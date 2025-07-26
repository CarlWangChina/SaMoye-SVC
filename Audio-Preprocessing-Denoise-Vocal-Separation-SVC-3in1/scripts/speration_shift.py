import torch
import torchaudio
import demucs.pretrained
import demucs.audio
import demucs.apply
import numpy as np
import librosa
from pydub import AudioSegment
from io import BytesIO
import soundfile as sf

class DemuxExtractor:
    def __init__(self, demucs_models: str = "htdemucs", device: str = "cuda"):
        self.demucs_models = demucs.pretrained.get_model(demucs_models).to(device)
        self.demucs_sampling_rate = self.demucs_models.samplerate
        self.demucs_num_channels = self.demucs_models.audio_channels
        self.device = device

    DEMUCS_DRUMS_INDEX = 0
    DEMUCS_BASS_INDEX = 1
    DEMUCS_OTHER_INDEX = 2
    DEMUCS_VOCAL_INDEX = 3

    @torch.inference_mode()
    def process_data(self, audio, sample_rate):
        waveform_resampled = demucs.audio.convert_audio(
            audio, sample_rate, self.demucs_sampling_rate, self.demucs_num_channels
        ).reshape(1, 2, -1)

        stems = demucs.apply.apply_model(
            self.demucs_models, waveform_resampled.to(self.device)
        ).to("cpu")[0]

        return stems

def pitch_shift(data, sample_rate, n_steps):
    # Apply pitch shift to each channel separately
    shifted = np.array([
        librosa.effects.pitch_shift(data[channel], sample_rate, n_steps)
        for channel in range(data.shape[0])
    ])
    return shifted

def audiosegment_from_array(data, sample_rate):
    buffer = BytesIO()
    sf.write(buffer, data.T, sample_rate, format='wav')  # Transpose to match shape (samples, channels)
    buffer.seek(0)
    return AudioSegment.from_file(buffer, format='wav')

def main(input_audio, output_audio, n_steps):
    # Step 1: Load input audio
    audio, sample_rate = torchaudio.load(input_audio)
    
    # Step 2: Separate tracks using DemuxExtractor
    demux_extractor = DemuxExtractor(device="cuda")
    stems = demux_extractor.process_data(audio, sample_rate)
    
    # Step 3: Pitch shift tracks (except drums)
    vocals_shifted = pitch_shift(stems[DemuxExtractor.DEMUCS_VOCAL_INDEX].numpy(), demux_extractor.demucs_sampling_rate, n_steps)
    bass_shifted = pitch_shift(stems[DemuxExtractor.DEMUCS_BASS_INDEX].numpy(), demux_extractor.demucs_sampling_rate, n_steps)
    other_shifted = pitch_shift(stems[DemuxExtractor.DEMUCS_OTHER_INDEX].numpy(), demux_extractor.demucs_sampling_rate, n_steps)

    # Step 4: Convert numpy arrays to AudioSegment
    drums_segment = audiosegment_from_array(stems[DemuxExtractor.DEMUCS_DRUMS_INDEX].numpy(), demux_extractor.demucs_sampling_rate)
    vocals_segment = audiosegment_from_array(vocals_shifted, demux_extractor.demucs_sampling_rate)
    bass_segment = audiosegment_from_array(bass_shifted, demux_extractor.demucs_sampling_rate)
    other_segment = audiosegment_from_array(other_shifted, demux_extractor.demucs_sampling_rate)

    # Step 5: Merge tracks
    combined = drums_segment.overlay(bass_segment).overlay(other_segment).overlay(vocals_segment)

    # Step 6: Export the final audio
    combined.export(output_audio, format="mp3")
    print(f"Final output saved to {output_audio}")

if __name__ == "__main__":
    input_audio = "/app/data/原调.mp3"  # Path to input audio file
    output_audio = "/app/data/变调.mp3"  # Path to save the final output audio file
    n_steps = 4  # Number of semitones to shift

    main(input_audio, output_audio, n_steps)
