from src.params import song_ids
import pretty_midi
import src.read_file as rf
import json
import librosa
import src.midi_lyric_align_helper_func as helper
import os
from fastapi import FastAPI, HTTPException
import asyncio
import argparse

import logging
from logging.handlers import RotatingFileHandler

# 创建一个logger
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler("my_app.log", maxBytes=1024 * 1024, backupCount=5),
        logging.StreamHandler(),
    ],
)


def change_note(song_id, test_name, test_midi_file):
    midi_file = f"data/midi/{test_midi_file}/{song_id}_new.mid"
    # Read the midi file
    midi_data = pretty_midi.PrettyMIDI(midi_file)

    origin_ds_file = f"data/ds/{test_name}/{song_id}.ds"
    new_ds_file = f"data/ds/{test_midi_file}/{song_id}.ds"
    os.makedirs(os.path.dirname(new_ds_file), exist_ok=True)
    # json 文件读取获取ds
    with open(origin_ds_file, "r") as f:
        ds = json.load(f)

    for sentence in ds:
        note_seq = sentence["note_seq"].split()
        offset = float(sentence["offset"])
        note_dur = helper.word_dur_to_seconds(sentence["note_dur"])
        note_start = offset
        # 遍历note_seq,找对应note
        for i, (note, note_dur) in enumerate(zip(note_seq, note_dur)):
            note_end = note_start + note_dur
            # 找到对应note
            if note != "rest":
                note_avail = []
                for instrument in midi_data.instruments:
                    if instrument.is_drum or instrument.program != 0:
                        continue
                    for note in instrument.notes:
                        if (
                            (note.start >= note_start and note.start <= note_end)
                            or (note.end >= note_start and note.end <= note_end)
                            or (note.start <= note_start and note.end >= note_end)
                            or note.end == instrument.notes[-1].end
                        ):
                            note_avail.append(note)
                        if note.start > note_end:
                            note_avail.append(note)
                            break

                note_sorted = sorted(
                    note_avail,
                    key=lambda x: helper.get_note_word_ratio(
                        x.start, x.end, note_start, note_end
                    ),
                    reverse=True,
                )
                note_closest = note_sorted[0]
                note_seq[i] = librosa.midi_to_note(note_closest.pitch)
                note_start = note_end
        sentence["note_seq"] = " ".join(note_seq)

    with open(new_ds_file, "w") as f:
        json.dump(ds, f, ensure_ascii=False, indent=4)
    # print(f"Note changed for {song_id} in {test_name}")


def change_ds_midi(test_inputs):
    # print(len(song_ids['singer600']))
    error_song_ids = []
    for song_id in song_ids[test_inputs.test_name]:
        try:
            new_ds_file = f"data/ds/{test_inputs.test_midi_file}/{song_id}.ds"
            if not os.path.exists(new_ds_file) or test_inputs.overwrite == 1:
                change_note(
                    str(song_id), test_inputs.test_name, test_inputs.test_midi_file
                )
        except:
            error_song_ids.append(song_id)

    print(
        f"Finsh changing {len(song_ids[test_inputs.test_name]) - len(error_song_ids)} songs, {len(error_song_ids)} songs failed."
    )
    if error_song_ids != [] and len(error_song_ids) < len(
        song_ids[test_inputs.test_name]
    ):
        logging.error(
            f"change_ds_midi for test {test_inputs.test_midi_file} failed, error num {len(error_song_ids)} error: {error_song_ids}"
        )
    else:
        logging.info(f"change_ds_midi for test {test_inputs.test_midi_file} successful")


if __name__ == "__main__":
    # 设置根目录
    root_dir = "/export/data/home/john/MuerSinger2/"
    # 更改工作目录到根目录
    os.chdir(root_dir)
    # 创建解析器
    parser = argparse.ArgumentParser()

    # 添加参数
    parser.add_argument("--test_name", type=str, default="test1", help="测试名称")
    parser.add_argument(
        "--test_midi_file", type=str, default="test1", help="测试 MIDI 文件路径"
    )
    parser.add_argument("--overwrite", type=int, default=1, help="是否覆盖已有文件")
    args = parser.parse_args()

    class TestInput:
        def __init__(self, test_name, test_midi_file, overwrite=1):
            self.test_name = test_name
            self.test_midi_file = test_midi_file
            self.overwrite = overwrite

    test_inputs = TestInput(args.test_name, args.test_midi_file, args.overwrite)
    change_ds_midi(test_inputs)
