import librosa
import os
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from src.params import song_ids
import argparse
import logging
from logging.handlers import RotatingFileHandler

# 创建一个logger  
logger = logging.getLogger(__name__)  
  
logging.basicConfig(level=logging.INFO,  
                    format='%(asctime)s - %(levelname)s - %(message)s',  
                    handlers=[RotatingFileHandler('my_app.log', maxBytes=1024*1024, backupCount=5),  
                              logging.StreamHandler()])
def mix_songs(song_id,test_name, spk_names):
    for spk_name in spk_names:
        vocal_path = f"/export/data/home/john/MuerSinger2/data/vocal/{test_name}/{spk_name}"
        companion_path = "/export/data/home/john/MuerSinger2/separated/mdx_extra"
        mix_path = f"/export/data/home/john/MuerSinger2/data/mix/{test_name}/{spk_name}"
        
        wav_name = f"{song_id}.wav"
        output_path = os.path.join(mix_path,wav_name)
        # 已经存在就不混了
        if os.path.exists(output_path):
            continue
        
        companion_wav_name = f"{song_id}/no_vocals.wav"
        # 载入两个音频文件
        assert os.path.exists(os.path.join(vocal_path,wav_name)), (f"Warning: {wav_name} not exists.")
            
        vocal_audio, vocal_sr = librosa.load(os.path.join(vocal_path,wav_name), sr=None)  # 假设音频1的采样率与音频2相同
        assert os.path.exists(os.path.join(companion_path,companion_wav_name)), print(f"Warning: {companion_wav_name} not exists.")
            
        companion_audio, companion_sr = librosa.load(os.path.join(companion_path,companion_wav_name), sr=None)  # 假设音频2的采样率与音频1相同


        # 将两个音频调整为相同的采样率
        if vocal_sr != companion_sr:
            print("Warning: Sample rates are different. Resampling companion audio to match vocal audio.")
            companion_audio = librosa.resample(companion_audio, companion_sr, vocal_sr)

        # 确保两个音频长度相同，如果不同可以进行裁剪或填充
        max_length = max(len(vocal_audio), len(companion_audio))
        vocal_audio = np.pad(vocal_audio, (0, max_length - len(vocal_audio)), mode='constant')
        companion_audio = np.pad(companion_audio, (0, max_length - len(companion_audio)), mode='constant')

        # 进行响度匹配
        # measure the loudness first 
        # meter = pyln.Meter(vocal_sr) # create BS.1770 meter
        # loudness = meter.integrated_loudness(companion_audio)

        # loudness normalize audio to -12 dB LUFS
        # loudness_normalized_companion_audio = pyln.normalize.loudness(companion_audio, loudness, -12.0)
        # loudness_normalized_vocal_audio = pyln.normalize.loudness(vocal_audio, loudness, -12.0)

        # 将两个音频混合
        mixed_audio = 1 * vocal_audio + 0.3 * companion_audio

        # 保存混合后的音频文件
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # sf保存为MP3
        sf.write(output_path, mixed_audio, vocal_sr, 'PCM_24')
        print("Mixed audio saved to:", output_path)

from pydub import AudioSegment

def mix_songs_as_mp3(song_id, test_name, spk_names):
    for spk_name in spk_names:
        vocal_path = f"/export/data/home/john/MuerSinger2/data/vocal/{test_name}/{spk_name}"
        companion_path = "/export/data/home/john/MuerSinger2/separated/mdx_extra"
        mix_path = f"/export/data/home/john/MuerSinger2/data/mix/{test_name}/{spk_name}"

        wav_name = f"{song_id}.wav"
        mp3_name = f"{song_id}.mp3"

        output_path = os.path.join(mix_path, mp3_name)

        # 已经存在就不混了
        if os.path.exists(output_path):
            continue

        companion_wav_name = f"{song_id}/no_vocals.wav"

        # 载入两个音频文件
        vocal_audio = AudioSegment.from_wav(os.path.join(vocal_path, wav_name))
        companion_audio = AudioSegment.from_wav(os.path.join(companion_path, companion_wav_name))

        # Adjust sample rates if needed (assuming they are already same)
        vocal_sr = vocal_audio.frame_rate
        companion_audio = companion_audio.set_frame_rate(vocal_sr)

        # Make durations equal
        max_duration = max(len(vocal_audio), len(companion_audio))
        vocal_audio = vocal_audio.set_channels(1).set_sample_width(2).set_frame_rate(vocal_sr)[:max_duration]
        companion_audio = companion_audio.set_channels(1).set_sample_width(2).set_frame_rate(vocal_sr)[:max_duration]

        # Match loudness and mix
        mixed_audio = vocal_audio.overlay(companion_audio - 10, position=0)

        # Save mixed audio
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        mixed_audio.export(output_path, format="mp3", bitrate="32k")

        print("Mixed audio saved to:", output_path)

def remix_main(test_input):
    false_songID = []
    _song_ids = song_ids[test_input.test_name]
    # _song_ids = [523572, 903269, 1169589, 1686672, 153343, 973273, 847575, 1303121, 888509, 880101, 122844, 1287594]
    for song_id in _song_ids:
        try:
            mix_songs_as_mp3(str(song_id), test_input.test_midi_file, test_input.spk)
        except:
            false_songID.append(song_id)
    
    # 打印出无法混合的歌曲ID
    raise ValueError(f'Total {len(false_songID)} songs failed: {false_songID}')

if __name__ == '__main__':
    false_songID = []

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name', type=str, default='test1', help='测试名称')
    parser.add_argument('--test_midi_file', type=str, default='test1', help='测试 MIDI 文件路径')
    parser.add_argument('--spk_name', type=list, default=['cpop_female'],help='歌手名称')

    args = parser.parse_args()
    _song_ids = song_ids[args.test_name]
    # _song_ids = [523572, 903269, 1169589, 1686672, 153343, 973273, 847575, 1303121, 888509, 880101, 122844, 1287594]
    for song_id in _song_ids:
        try:
            mix_songs_as_mp3(str(song_id), args.test_midi_file, args.spk_name)
        except:
            false_songID.append(song_id)
    
    # 打印出无法混合的歌曲ID
    if len(false_songID) > 0:
        logging.error(f"mix_song for test {args.test_midi_file} failed, error num {len(false_songID)} error: {false_songID}")
    else:
        logging.info(f"mix_song for test {args.test_midi_file} successful")