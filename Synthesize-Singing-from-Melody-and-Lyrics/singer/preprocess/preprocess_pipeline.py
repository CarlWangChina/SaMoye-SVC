"""
This module contains the main processing pipeline for the singer project.
"""

import os
import json
from pathlib import Path
from singer.utils.logging import get_logger
from .change_lyric import mq_change_lyric, mq_clean_lyric
from .midi_lyric_align import mq_midi_lyric_align
from .change_ds_midi import mq_change_note
from .generate_vocal import mq_run_diffsinger_command
from .utils.wav_2_mp3 import convert_wav_to_mp3

logger = get_logger(__name__)

# 设置根目录
# ROOT_DIR = "/export/data/home/john/DuiniutanqinSinger/"


def clean_lyric(time_json):
    """
    Clean the lyrics in the JSON data.

    Args:
        time_json (dict): The time JSON data.

    Returns:
        dict: The new time JSON data.

    """
    # 1.0 清洗数据，只保留C段
    new_time_json = mq_clean_lyric(time_json)
    return new_time_json


def change_lyric(time_json, lyricoss):
    """
    Change the lyrics in the JSON data.

    Args:
        time_json (dict): The time JSON data.
        lyricoss (str): The new lyrics.

    Returns:
        dict: The new time JSON data.

    """
    # 1.1 将json中的歌词换成新的歌词
    new_time_json = mq_change_lyric(time_json, lyricoss)
    return new_time_json


def midi_lyric_align(song_id, midi, new_time_json, for_ace=False):
    """
    Align the MIDI file with the lyrics.

    Args:
        song_id (str): The song ID.
        midi (str): The MIDI file path.
        new_midi (str): The new MIDI file path.
        new_time_json (dict): The new time JSON data.

    Returns:
        dict: The DS data.
    """
    # 1.2 将 json 中的时间戳、cccr转换为 ds 文件
    ds_data = mq_midi_lyric_align(str(song_id), midi, new_time_json, for_ace)
    return ds_data


def change_note(song_id, ds_data, midi):
    """
    Change the note of the DS file based on the MIDI file.

    Args:
        song_id (str): The song ID.
        ds_data (dict): The DS data.
        midi (str): The MIDI file path.

    Returns:
        dict: The new DS data with the notes changed.

    """
    # 2. 根据这个 midi chang_ds_midi 为相应midi的 ds
    ds_data_new = mq_change_note(str(song_id), ds_data, midi)
    return ds_data_new


def process_ds_and_generate_vocal(ds_temp_path, ds_temp_output_path, vocal_temp_path):
    """
    Process the DS file and generate the vocal audio file.

    Args:
        ds_temp_path (Path): The path to the DS file.
        ds_temp_output_path (Path): The path to the output DS file.
        vocal_temp_path (Path): The path to the generated vocal audio file.

    Returns:
        None

    """
    # 3. 根据这个 ds 调用 variance 模型进行推理
    # 4. 调用acoustic模型进行推理
    mq_run_diffsinger_command(
        ds_input_path=ds_temp_path,
        ds_output_path=ds_temp_output_path,  # Assuming the output path is the same as the input path for simplicity
        vocal_path=vocal_temp_path,
    )


def convert_audio_format(song_id, vocal_temp_path, wav_name=None):
    """Convert the generated vocal audio file to MP3 format.

    Args:
        song_id (str): The song ID.
        vocal_temp_path (Path): The path to the generated vocal audio file.

    Returns:
        None
    """
    # 5. 转换音频格式
    vocal_wav_path = (
        vocal_temp_path / Path(f"{song_id}.wav")
        if wav_name is None
        else vocal_temp_path / Path(f"{wav_name}.wav")
    )
    vocal_mp3_path = (
        vocal_temp_path / Path(f"{song_id}.mp3")
        if wav_name is None
        else vocal_temp_path / Path(f"{wav_name}.mp3")
    )
    convert_wav_to_mp3(vocal_wav_path, vocal_mp3_path)


def preprocess_main(song_id, time_json, lyricoss, midi, new_midi, root_path):
    """
    Main processing pipeline for the singer project.

    Args:
        song_id (str): The song ID.
        time_json (dict): The time JSON data.
        lyricoss (str): The new lyrics.
        midi (str): The MIDI file path.
        new_midi (str): The new MIDI file path.
        root_path (str): The root path of the project.

    Returns:
        Path: The path to the generated vocal audio file.
    """
    # 更改工作目录到根目录
    os.chdir(root_path)

    # 0. 清洗数据，只保留C段
    time_json_modified = clean_lyric(time_json)
    # 1. 歌词更改
    if lyricoss is not None:
        new_time_json = change_lyric(time_json_modified, lyricoss)
    else:
        new_time_json = time_json_modified

    # 2. MIDI和歌词对齐
    ds_data = midi_lyric_align(song_id, midi, new_time_json)

    # 3. MIDI和DS文件处理
    ds_data_new = change_note(song_id, ds_data, new_midi)

    # 4. 创建文件夹
    ds_temp_path = root_path / Path(f"data/ds/temp/input/{song_id}.ds")
    ds_temp_output_path = root_path / Path(f"data/ds/temp/output/{song_id}.ds")
    vocal_temp_path = root_path / Path("data/vocal/temp/")

    # 确保文件夹存在
    ds_temp_path.parent.mkdir(parents=True, exist_ok=True)

    # 5. 写入DS数据
    with open(ds_temp_path, "w", encoding="utf-8") as f:
        json.dump(ds_data_new, f, ensure_ascii=False, indent=4)

    # 6. 处理DS文件并生成vocal
    process_ds_and_generate_vocal(
        ds_temp_path,
        ds_temp_output_path,
        vocal_temp_path,
    )

    # 7. 转换音频格式
    convert_audio_format(song_id, vocal_temp_path)

    return vocal_temp_path


# if __name__ == "__main__":
#     # 假设我们有一些输入数据
#     song_id = "123"
#     time_json = {"timing": "data"}
#     lyricoss = "new lyrics"
#     midi = "path/to/midi/file"
#     new_midi = "path/to/new/midi/file"
#     root_path = ROOT_DIR

#     # 运行主处理函数
#     preprocess_main(song_id, time_json, lyricoss, midi, new_midi, root_path)
