"""
    change ds file to aces file
"""

import json
from pathlib import Path
from pypinyin import lazy_pinyin
import librosa

from .ACE_phonemes import main as ace_ph


class NoteClass:
    """Note class for ACE file"""

    def __init__(
        self,
        start_time,
        end_time,
        type="general",
        pitch=None,
        language="ch",
        phone=None,
    ):
        self.start_time = start_time
        self.end_time = end_time
        self.type = type
        self.pitch = pitch
        self.language = language
        self.phone = phone

    def return_dict(self):
        """Return note as a dictionary"""
        note = {}
        note["start_time"] = self.start_time
        note["end_time"] = self.end_time
        if self.type:
            note["type"] = self.type
        if self.pitch:
            note["pitch"] = self.pitch
        if self.language:
            note["language"] = self.language
        if self.phone:
            note["phone"] = self.phone
        return note


def ds_to_aces(ds_path, ace_dir_path):
    """
    Process ds file to aces file

    Args:
        ds_path (Path): path to the original ds file
        ace_dir_path (Path): path to save the aces file

    Returns:
        None

    """
    with open(ds_path, "r", encoding="utf-8") as f:
        ds = json.load(f)

    aces = {}
    aces["offset"] = ds[0]["offset"]
    aces["version"] = 1.1
    aces["lyrics"] = []
    aces["notes"] = []
    offset = 0

    for ds_i, sequence in enumerate(ds):
        word_seq = sequence["word_seq"].split()
        word_string = "".join([w for w in word_seq if w not in ("AP", "SP")])
        pinyin_seq = lazy_pinyin(word_string)

        word_dur = [float(d) for d in sequence["word_dur"].split()]
        note_seq = sequence["note_seq"].split()
        aces["lyrics"].append(sequence["word_seq"])

        # 生成notes，根据word_seq中的AP和SP来生成br和sp，其他的生成音符。TODO：每一个句子长于18s怎么办？是在midi_lyric_align中处理还是在这里处理？
        assert (
            sum(word_dur) <= 18
        ), f"Duration of the sentence is longer than 18s: {sum(word_dur)}"
        for i, word in enumerate(word_seq):
            if word == "AP":
                note = NoteClass(offset, offset + word_dur[i], type="br")
                offset += word_dur[i]
            elif word == "SP":
                note = NoteClass(offset, offset + word_dur[i], type="sp")
                offset += word_dur[i]
            else:
                pinyin = pinyin_seq.pop(0)
                pitch = librosa.note_to_midi(note_seq[i])
                note = NoteClass(
                    offset,
                    offset + word_dur[i],
                    pitch=pitch,
                    phone=ace_ph.pinyin_to_phoneme(pinyin),
                )
                offset += word_dur[i]
            aces["notes"].append(note.return_dict())

        if ds_i != len(ds) - 1:
            next_end = ds[ds_i + 1]["offset"] + sum(
                [float(d) for d in ds[ds_i + 1]["word_dur"].split()]
            )
        else:
            next_end = sequence["offset"] + sum(
                [float(d) for d in sequence["word_dur"].split()]
            )

        this_end = sequence["offset"] + sum(word_dur)
        next_duration = next_end - aces["offset"]
        this_duration = this_end - aces["offset"]

        # 预算下一个加起来会不会大于18s，或者这个句子已经大于17s，或者是最后一个句子，就保存为一个文件
        if next_duration > 18 or this_duration > 17 or ds_i == len(ds) - 1:
            ace_dir_path.mkdir(exist_ok=True)
            s_i = 1
            ace_file_path = ace_dir_path / f"{ds_path.stem}_sentence_{s_i}.aces"
            while ace_file_path.exists():
                s_i += 1
                ace_file_path = ace_dir_path / f"{ds_path.stem}_sentence_{s_i}.aces"

            with open(ace_file_path, "w", encoding="utf-8") as f:
                json.dump(aces, f, ensure_ascii=False, indent=4)
                print(f"Saved to {ace_file_path}")

            if ds_i != len(ds) - 1:
                aces = {}
                aces["offset"] = ds[ds_i + 1]["offset"]
                aces["version"] = 1.1
                aces["lyrics"] = []
                aces["notes"] = []
                offset = 0
        else:
            if ds_i != len(ds) - 1:
                offset = ds[ds_i + 1]["offset"] - aces["offset"]


if __name__ == "__main__":
    # Usage example:
    origin_ds_file = Path("/home/john/DuiniutanqinSinger/data/ds/temp/1686672.ds")
    ace_path = Path(
        f"/home/john/DuiniutanqinSinger/data/aces/temp/{origin_ds_file.stem}"
    )

    ace_path.mkdir(parents=True, exist_ok=True)
    # 如果acepath已经存在，清空原有文件
    for ace_file in ace_path.glob("*.aces"):
        ace_file.unlink()
    ds_to_aces(origin_ds_file, ace_path)
