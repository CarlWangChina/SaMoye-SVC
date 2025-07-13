import os
import json
from pydub import AudioSegment
from pathlib import Path

def combine_audio(song_id, ogg_folder:Path, aces_folder:Path, combined_folder:Path):
    """ Combine multiple ogg files into a single audio file

    Args:
        ogg_folder (Path): path to the folder containing ogg files
        aces_folder (Path): path to the folder containing aces files
        combined_folder (Path): path to save the combined audio file

    Returns:
        None

    """

    # 读取所有的ogg文件并按名称排序
    ogg_files = sorted(ogg_folder.glob("*.ogg"), key=lambda x: int(x.stem.split("_")[-1]))

    # 创建一个空的音频对象
    combined_audio = AudioSegment.empty()

    # 从每个ogg文件开始，读取并合并音频
    for ogg_file in ogg_files:
        # 读取aces文件获取偏移量
        with open(aces_folder / (ogg_file.stem + '.aces'), encoding='utf-8') as f:
            aces_data = json.load(f)
            offset = aces_data['offset']

        # 读取ogg文件
        audio_segment = AudioSegment.from_ogg(ogg_file)

        # 计算需要添加的静音片段的长度（以毫秒为单位）
        silence_length = int(offset * 1000)
        
        # 创建静音片段
        silence_segment = AudioSegment.silent(duration=silence_length)
        
        # 将静音片段与当前ogg文件混合
        audio_segment_with_silence = silence_segment + audio_segment
        
        # 将混合后的音频直接加到合并后的音频对象上
        combined_audio = audio_segment_with_silence.overlay(combined_audio)

    # 将合并后的音频保存为ogg文件
    combined_folder.mkdir(parents=True, exist_ok=True)
    combined_audio_path = combined_folder / f"{song_id}.wav"
    combined_audio.export(combined_audio_path, format='wav')
    # combined_path = combined_folder / f"{song_id}.mp3"
    # combined_audio.export(combined_path, format='mp3', bitrate='32k')

if __name__ == "__main__":
    song_id = '1686672'
    # Example usage:
    audio_path = Path("/home/john/DuiniutanqinSinger/data/audio_ace/temp/1686672")
    ace_path = Path(f"/home/john/DuiniutanqinSinger/data/aces/temp/1686672")
    combined_path = Path('/home/john/DuiniutanqinSinger/data/combined/temp')
    # combined_path.mkdir(parents=True, exist_ok=True)
    combine_audio(song_id, audio_path, ace_path, combined_path)
    # python convert_ds.py csv2ds /home/john/DuiniutanqinSinger/data/combined/temp/transcriptions/transcriptions.csv /home/john/DuiniutanqinSinger/data/combined/temp --pe rmvpe
