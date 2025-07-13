""" 从歌词文件中提取音符序列 """

# -*- coding: utf-8 -*-
import os
import logging
from logging.handlers import RotatingFileHandler

import librosa
import cn2an

# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from .utils import lyric_to_phoneme as l2p
from .utils import midi_lyric_align_helper_func as helper
from .utils import read_file as rf


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


AP_TIME = 0.3
AP_MIN_TIME = 0.1
SP_MIN_TIME = 0.1
SINGING_LONGEST_TIME = 1


# 读取列表中每个字典的信息，并生成音符序列
def generate_notes(data, midi_data, pinyin_phoneme_dict, min_note_dur):
    """
    生成音符序列

    Args:
        data: JSON数据
        midi_data: MIDI数据
        pinyin_phoneme_dict: 拼音音素字典
        min_note_dur: 最小音符时长

    Returns:
        word_note_data: 音符序列

    """

    word_note_data = []

    last_d_i = len(data) - 1
    for d_i, record in enumerate(data):
        note_seq = []
        note_dur = []
        note_slur = []
        word_dur_new = []
        word_seq_new = []
        origin_offset = float(record["start"]) / 1000
        note_offset = -1
        # print(f"Processing {record}")
        # 提取音符信息
        word_seq, word_dur, word_start, word_end = helper.words_seperate(
            record["words"]
        )

        if word_seq == [""]:  # 去除空句子
            continue
        for i, word in enumerate(
            word_seq
        ):  # 去除标识多人唱歌手的符号之前的内容，如：'男:', '孩子:'
            if word == ":" or word == "：":
                while i >= 0:
                    word_seq[i] = "SP"
                    i -= 1
            if word == "":
                word_seq[i] = "SP"

        last_w_i = len(word_seq) - 1
        for w_i, (word, dur, start_time, end_time) in enumerate(
            zip(word_seq, word_dur, word_start, word_end)
        ):
            if not word_seq_new and (word == "SP" or word == "AP"):
                origin_offset += dur
                continue
            if w_i == last_w_i and d_i != last_d_i:
                if (
                    end_time > float(data[d_i + 1]["start"]) / 1000
                    and start_time < float(data[d_i + 1]["start"]) / 1000
                ):
                    end_time = float(data[d_i + 1]["start"]) / 1000
                assert (
                    start_time <= end_time
                ), f"start_time {start_time} >= end_time {end_time}"

            note_avail = []

            # 非中文：阿拉伯数字转换为相应的中文数字，然后英文先变为SP、rest
            word_type = l2p._get_char_type(word)
            if word_type != "chinese":
                if word_type == "en_digit":
                    word = cn2an.an2cn(word, "low")
                elif word_seq_new:
                    word_seq_new.append(word)
                    word_dur_new.append(dur)
                    note_seq.append("rest")
                    note_dur.append(dur)
                    note_slur.append(0)
                    continue
                else:
                    # 句子开始的英文，直接跳过, 但是要记录时间
                    origin_offset += dur
                    continue

            # 将时间范围内的音符放入候选名单，另外，为了防止时间戳太短导致音符数量不够，还把其下一个音符暂时纳入
            for instrument in midi_data.instruments:
                if instrument.is_drum or instrument.program != 0:
                    continue
                for note in instrument.notes:
                    if (
                        (note.start >= start_time and note.start <= end_time)
                        or (
                            note.end >= start_time + min_note_dur / 2
                            and note.end <= end_time
                        )
                        or (note.start <= start_time and note.end >= end_time)
                        or note.end == instrument.notes[-1].end
                    ):
                        note_avail.append(note)
                    if note.start > end_time:
                        note_avail.append(note)
                        break

            # 排序 按照音符的开始时间离start_time的距离
            note_sorted = sorted(
                note_avail,
                key=lambda x, st=start_time, et=end_time: abs(x.start - st)
                + abs(x.end - et),
            )
            # note_sorted = sorted(note_avail, key=lambda x: helper.get_note_word_ratio(x.start, x.end, start_time, end_time))
            note_closest = note_sorted[0]

            slur = 0
            note_time_sum = 0
            note_time_sum += note_closest.end - note_closest.start

            # threshold 为音符时长已满足字时长的门槛
            threshold = 0

            # while note_time_sum < threshold and len(note_sorted) > 1:
            #     note_sorted.pop(0)
            #     note_closest = note_sorted[0]
            #     note_time_sum = note_closest.end - note_closest.start

            # 开始处理音符
            if (
                note_time_sum >= threshold
            ):  # 最接近的 note 约等于 word duration 直接加入
                # 3-27 test1：最简单的一字一音，每个字取原时间戳
                final_dur = dur  # if w_i != last_w_i else note_time_sum # 最后一个字的音符时长取最长, 其他的取字的时长

                # 处理中文数字为多个的情况
                if len(word) > 1:
                    for char in word:
                        word_seq_new.append(char)
                        word_dur_new.append(final_dur / len(word))
                else:
                    word_seq_new.append(word)
                    word_dur_new.append(final_dur)

                note_dur.append(final_dur)
                note_name = librosa.midi_to_note(note_closest.pitch)
                note_seq.append(note_name)
                note_slur.append(slur)
                if note_offset < 0:
                    note_offset = note_closest.start

                # 当音符时长超过最大唱歌时长时，需要将超过部分设为空白
                if final_dur > SINGING_LONGEST_TIME:
                    over_time = final_dur - SINGING_LONGEST_TIME

                    if over_time > SP_MIN_TIME + AP_TIME:
                        word_dur_new[-1] = SINGING_LONGEST_TIME
                        note_dur[-1] = SINGING_LONGEST_TIME
                        # 非最后一个需要补充SP+AP，其中SP的时长为over_time - AP_MIN_TIME，AP的时长为AP_MIN_TIME
                        if w_i != last_w_i:
                            word_seq_new.append("SP")
                            word_dur_new.append(over_time - AP_TIME)
                            note_seq.append("rest")
                            note_dur.append(over_time - AP_TIME)
                            note_slur.append(0)
                            word_seq_new.append("AP")
                            word_dur_new.append(AP_TIME)
                            note_seq.append("rest")
                            note_dur.append(AP_TIME)
                            note_slur.append(0)
                    elif over_time >= AP_MIN_TIME:
                        word_dur_new[-1] = SINGING_LONGEST_TIME
                        note_dur[-1] = SINGING_LONGEST_TIME
                        # 非最后一个需要补充AP
                        if w_i != last_w_i:
                            word_seq_new.append("AP")
                            word_dur_new.append(over_time)
                            note_seq.append("rest")
                            note_dur.append(over_time)
                            note_slur.append(0)
            else:
                raise ValueError(
                    f"note_time_sum {note_time_sum} < threshold {threshold}"
                )

            # 两个字之间有空隙，补充AP SP rest
            if (
                w_i != last_w_i
                and end_time != word_end[-1]
                and end_time != word_start[w_i + 1]
            ):
                gap = word_start[w_i + 1] - start_time - dur
                if gap > 0:  # 有空隙
                    if gap > AP_TIME:  # 有空隙，且大于AP_time
                        sp_time = gap - AP_TIME
                        if (
                            sp_time >= SP_MIN_TIME
                        ):  # sp_time 大于 SP_MIN_TIME 才需要补充SP
                            note_seq.append("rest")
                            note_dur.append(sp_time)
                            note_slur.append(0)
                            word_seq_new.append("SP")
                            word_dur_new.append(sp_time)
                        else:  # sp_time 小于 SP_min_time，加到上一个音符上
                            note_dur[-1] += sp_time
                            word_dur_new[-1] += sp_time

                        note_seq.append("rest")
                        note_dur.append(AP_TIME)
                        note_slur.append(0)
                        word_seq_new.append("AP")
                        word_dur_new.append(AP_TIME)
                    elif gap >= AP_MIN_TIME:  # 有空隙，且大于AP_min_time
                        note_seq.append("rest")
                        note_dur.append(gap)
                        note_slur.append(0)
                        word_seq_new.append("AP")
                        word_dur_new.append(gap)
                    else:  # 有空隙，但小于AP_min_time，加到上一个音符上
                        note_dur[-1] += gap
                        word_dur_new[-1] += gap

        ph_seq = l2p.word_to_phoneme(word_seq_new, pinyin_phoneme_dict)
        if ph_seq == "":
            continue

        word_note = {}
        # word_note['offset'] = float(record['start'])/1000

        # offset 矫正，若是差太远那就不能够矫正
        word_note["offset"] = (
            note_offset
            if abs(origin_offset - note_offset) < 2 * min_note_dur
            else origin_offset
        )
        word_note["word_seq"] = " ".join(word_seq_new)
        word_note["word_dur"] = " ".join(map(str, word_dur_new))
        word_note["ph_seq"] = ph_seq
        word_note["note_seq"] = " ".join(note_seq)
        word_note["note_dur"] = " ".join(map(str, note_dur))
        word_note["note_slur"] = " ".join(map(str, note_slur))

        word_note_data.append(word_note)

    return word_note_data


def merge_records(record1, record2, gap, min_note_dur):
    """
    合并两个记录

    Args:
        record1: 第一个记录
        record2: 第二个记录
        gap: 两个记录之间的时间差
        min_note_dur: 最小音符时长

    Returns:
        merged_record: 合并后的记录

    """

    if gap > AP_TIME + SP_MIN_TIME:
        merged_word_seq = record1["word_seq"] + " SP AP " + record2["word_seq"]
        merged_word_dur = (
            record1["word_dur"] + f" {gap - AP_TIME} {AP_TIME} " + record2["word_dur"]
        )
        merged_offset = record1["offset"]  # 使用第一个记录的offset作为合并后的offset
        merged_ph_seq = record1["ph_seq"] + " SP AP " + record2["ph_seq"]
        merged_note_seq = record1["note_seq"] + " rest rest " + record2["note_seq"]
        merged_note_dur = (
            record1["note_dur"] + f" {gap - AP_TIME} {AP_TIME} " + record2["note_dur"]
        )
        merged_note_slur = record1["note_slur"] + " 0 0 " + record2["note_slur"]
    elif gap > AP_MIN_TIME:
        merged_word_seq = record1["word_seq"] + " AP " + record2["word_seq"]
        merged_word_dur = record1["word_dur"] + f" {gap} " + record2["word_dur"]
        merged_offset = record1["offset"]
        merged_ph_seq = record1["ph_seq"] + " AP " + record2["ph_seq"]
        merged_note_seq = record1["note_seq"] + " rest " + record2["note_seq"]
        merged_note_dur = record1["note_dur"] + f" {gap} " + record2["note_dur"]
        merged_note_slur = record1["note_slur"] + " 0 " + record2["note_slur"]
    elif gap > 0:
        merged_word_seq = record1["word_seq"] + " " + record2["word_seq"]
        record1_word_dur = helper.word_dur_to_seconds(record1["word_dur"])
        record1_word_dur[-1] += gap
        record1["word_dur"] = " ".join(map(str, record1_word_dur))
        merged_word_dur = record1["word_dur"] + " " + record2["word_dur"]
        merged_offset = record1["offset"]
        merged_ph_seq = record1["ph_seq"] + " " + record2["ph_seq"]
        merged_note_seq = record1["note_seq"] + " " + record2["note_seq"]
        record1_note_dur = helper.word_dur_to_seconds(record1["note_dur"])
        record1_note_dur[-1] += gap
        record1["note_dur"] = " ".join(map(str, record1_note_dur))
        merged_note_dur = record1["note_dur"] + " " + record2["note_dur"]
        merged_note_slur = record1["note_slur"] + " " + record2["note_slur"]
    else:
        merged_word_seq = record1["word_seq"] + " " + record2["word_seq"]
        record1_word_dur = helper.word_dur_to_seconds(record1["word_dur"])
        # 按照比例分配gap的时间
        record1_word_dur_sum = sum(record1_word_dur)
        for i, dur in enumerate(record1_word_dur):
            record1_word_dur[i] += gap * dur / record1_word_dur_sum
            assert (
                record1_word_dur[i] >= 0
            ), f"record1_word_dur[{i}] {record1_word_dur[i]} < 0"

        record1["word_dur"] = " ".join(map(str, record1_word_dur))
        merged_word_dur = record1["word_dur"] + " " + record2["word_dur"]
        merged_offset = record1["offset"]
        merged_ph_seq = record1["ph_seq"] + " " + record2["ph_seq"]
        merged_note_seq = record1["note_seq"] + " " + record2["note_seq"]
        record1_note_dur = helper.word_dur_to_seconds(record1["note_dur"])
        # 按照比例分配gap的时间
        record1_note_dur_sum = sum(record1_note_dur)
        for i, dur in enumerate(record1_note_dur):
            record1_note_dur[i] += gap * dur / record1_note_dur_sum
            assert (
                record1_note_dur[i] >= 0
            ), f"record1_note_dur[{i}] {record1_note_dur[i]} < 0"
        record1["note_dur"] = " ".join(map(str, record1_note_dur))
        merged_note_dur = record1["note_dur"] + " " + record2["note_dur"]
        merged_note_slur = record1["note_slur"] + " " + record2["note_slur"]
    return {
        "offset": merged_offset,
        "word_seq": merged_word_seq,
        "word_dur": merged_word_dur,
        "ph_seq": merged_ph_seq,
        "note_seq": merged_note_seq,
        "note_dur": merged_note_dur,
        "note_slur": merged_note_slur,
    }


def change_record_time(record, gap):
    if gap < 0:
        record1_word_dur = helper.word_dur_to_seconds(record["word_dur"])
        # 按照比例分配gap的时间
        record1_word_dur_sum = sum(record1_word_dur)
        for i, dur in enumerate(record1_word_dur):
            record1_word_dur[i] += gap * dur / record1_word_dur_sum
            assert (
                record1_word_dur[i] >= 0
            ), f"record1_word_dur[{i}] {record1_word_dur[i]} < 0"

        record["word_dur"] = " ".join(map(str, record1_word_dur))
        merged_word_dur = record["word_dur"]

        record1_note_dur = helper.word_dur_to_seconds(record["note_dur"])
        # 按照比例分配gap的时间
        record1_note_dur_sum = sum(record1_note_dur)
        for i, dur in enumerate(record1_note_dur):
            record1_note_dur[i] += gap * dur / record1_note_dur_sum
            assert (
                record1_note_dur[i] >= 0
            ), f"record1_note_dur[{i}] {record1_note_dur[i]} < 0"
        record["note_dur"] = " ".join(map(str, record1_note_dur))
        merged_note_dur = record["note_dur"]
        return {
            "offset": record["offset"],
            "word_seq": record["word_seq"],
            "word_dur": merged_word_dur,
            "ph_seq": record["ph_seq"],
            "note_seq": record["note_seq"],
            "note_dur": merged_note_dur,
            "note_slur": record["note_slur"],
        }
    else:
        return record


def check_offsets(songID, data, min_note_dur):
    """
    检查记录的offsets

    Args:
        songID: 歌曲ID
        data: 数据
        min_note_dur: 最小音符时长

    Returns:
        data: 更新后的数据
    """
    # print(f"Checking offsets for {songID} {data}")
    if data[0]["offset"] >= AP_TIME:
        data[0] = helper.add_AP_front(data[0], AP_TIME)
    i = 0
    min_interval = AP_TIME
    while i < len(data):
        merged = False
        while i + 1 < len(data):
            current_record = data[i]
            next_record = data[i + 1]

            end_time = current_record["offset"] + sum(
                helper.word_dur_to_seconds(current_record["word_dur"])
            )

            # if end_time > next_record['offset']:
            #     print(f"{songID} offset {current_record['offset']} end_time {end_time} next_record {next_record['offset']} dis {end_time - next_record['offset']}")
            if end_time + min_interval > next_record["offset"]:
                # 合并当前记录和下一个记录
                merged_record = merge_records(
                    current_record,
                    next_record,
                    next_record["offset"] - end_time,
                    min_note_dur,
                )

                # 更新列表中的记录
                data[i] = merged_record
                del data[i + 1]

                # 标记为已合并
                merged = True
                break  # 跳出内部循环，重新检查合并后的记录与下一个记录的关系
            else:
                try:
                    if (
                        data[i + 1]["word_seq"][0] != "S"
                        and data[i + 1]["word_seq"][0] != "A"
                    ):
                        data[i + 1] = helper.add_AP_front(next_record, min_interval)
                except Exception as e:
                    print(f"index {i + 1} {data[i + 1]} {e}")
                # 不需要合并，退出内部循环
                break

        # 如果没有合并，或者已经合并并更新了列表，移动到下一个记录
        if not merged:
            i += 1

    return data


def check_offsets_for_ace(data):
    """
    检查记录的offsets,将AP和SP加到前面,保证每一句的长度不超过后一句的offset
    """
    if data[0]["offset"] > AP_TIME:
        data[0] = helper.add_AP_front(data[0], AP_TIME)
    i = 0
    min_interval = AP_MIN_TIME
    for i in range(len(data) - 2):
        current_record = data[i]
        next_record = data[i + 1]

        end_time = current_record["offset"] + sum(
            helper.word_dur_to_seconds(current_record["word_dur"])
        )

        if end_time > next_record["offset"] - min_interval:
            new_record = change_record_time(
                current_record, next_record["offset"] - end_time
            )
            data[i] = new_record
        else:
            if data[i + 1]["word_seq"][0] != "S" and data[i + 1]["word_seq"][0] != "A":
                data[i + 1] = helper.add_AP_front(next_record, min_interval)
    return data


def mq_midi_lyric_align(
    song_id: str, origin_midi, json_data, for_ace: bool = False
):
    """
    生成歌词对应的音符序列

    Args:
        song_id: 歌曲ID
        origin_midi: 原始MIDI文件
        json_data: JSON数据

    Returns:
        ds_data: 音符序列

    """
    dict_file = "singer/dictionaries/opencpop-extension.txt"
    pinyin_phoneme_dict = rf.read_dict(dict_file)

    min_note_dur = 60 / rf.get_bpm(origin_midi) / 8
    midi_data = origin_midi
    word_note_data = generate_notes(
        json_data, midi_data, pinyin_phoneme_dict, min_note_dur
    )

    if for_ace:
        merged_word_note_data = check_offsets_for_ace(word_note_data)
    else:
        merged_word_note_data = check_offsets(song_id, word_note_data, min_note_dur)

    ds_data = helper.mq_add_ph_num(merged_word_note_data, dictionary=dict_file)

    return ds_data
