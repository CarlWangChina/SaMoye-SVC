"""
    1. change json lyric
    2. change json to ds
    3. change ds's note
    4. split ds to aces
    5. aces to audio
    6. merge aces' audio
    7. change aces' audio to ds to vocal
"""

import json
import os
from pathlib import Path

from .preprocess.preprocess_pipeline import (
    change_lyric,
    change_note,
    midi_lyric_align,
    clean_lyric,
    convert_audio_format,
)
from .postprocess.test_postprocess_pipeline import postprocess_main


def pipline_aces2ds_main(song_id, time_json, lyricoss, midi, new_midi, root_path):
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
    ds_data = midi_lyric_align(song_id, midi, new_time_json, for_ace=True)

    # 3. MIDI和DS文件处理
    ds_data_new = change_note(song_id, ds_data, new_midi)

    # 4. 创建文件夹
    ds_temp_path = root_path / Path(f"data/ds/temp/ace_input/{song_id}.ds")

    # 确保文件夹存在
    ds_temp_path.parent.mkdir(parents=True, exist_ok=True)

    # 5. 写入DS数据
    with open(ds_temp_path, "w", encoding="utf-8") as f:
        json.dump(ds_data_new, f, ensure_ascii=False, indent=4)

    vocal_path = postprocess_main(song_id, root_path, ds_temp_path)
    
    convert_audio_format(song_id, vocal_path, wav_name=f"{song_id}_split")
    return vocal_path
