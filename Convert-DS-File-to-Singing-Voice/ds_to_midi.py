import json
import pretty_midi
import librosa
from src.params import song_ids
import argparse
import os

def json_to_midi(song_id, test_name, test_midi_file):
    ds_file = f'data/ds/{test_name}/{song_id}.ds'
    midi_file = f'data/bpm_midi/{test_name}/{song_id}_src.mp3_5b.mid'
    midi_for_lyric_file = f'data/midi_for_lyric/{test_name}/'
    os.makedirs(midi_for_lyric_file, exist_ok=True)
    # 读取 JSON 文件
    with open(ds_file, 'r') as f:
        data = json.load(f)

    original_midi = pretty_midi.PrettyMIDI(midi_file)

    # 创建 PrettyMIDI 对象
    midi = pretty_midi.PrettyMIDI()
    # 添加一个默认乐器（Instrument）
    instrument = pretty_midi.Instrument(program=0)  # 使用默认的 program（乐器编号）
    midi.instruments.append(instrument)

    # 遍历每个句子的数据
    last_d_i = len(data) - 1
    for d_i, sentence in enumerate(data):
        # 解析数据
        offset = sentence['offset']
        word_dur = list(map(float, sentence['word_dur'].split()))
        note_seq = sentence['note_seq'].split()
        note_dur = list(map(float, sentence['note_dur'].split()))

        # 计算句子的起始时间
        start_time = offset

        # 遍历每个单词和音符
        for w_i, (word_duration, note, note_duration) in enumerate(zip(word_dur, note_seq, note_dur)):
            if note == 'rest':
                if midi.instruments[0].get_end_time() != 0:
                    # 给前一个音加时间
                    midi.instruments[0].notes[-1].end = start_time + word_duration
            else:
                # 创建音符对象
                note_obj = pretty_midi.Note(
                    velocity=100,
                    pitch=librosa.note_to_midi(note),
                    start=start_time,
                    end=start_time + note_duration
                )
                # 将音符添加到 MIDI 对象中
                midi.instruments[0].notes.append(note_obj)

            # 更新下一个音符的起始时间
            start_time += word_duration

    # 将原始 MIDI 文件的和弦轨道添加到新的 MIDI 文件中
    midi.instruments.append(original_midi.instruments[1])
    midi.instruments[0].notes[-1].end = midi.instruments[1].notes[-1].end
    # 保存 MIDI 文件
    midi.write(os.path.join(midi_for_lyric_file, f'{song_id}.mid'))

def ds_to_midi(test_name, test_midi_file):
    for song_id in song_ids[test_name]:
        try:
            json_to_midi(str(song_id), test_name, test_midi_file)
        except:
            raise ValueError(f'Error occured when processing song {song_id}')

# 使用示例
if __name__ == '__main__':
    # 创建解析器
    parser = argparse.ArgumentParser()
    
    # 添加参数
    parser.add_argument('--test_name', type=str, default='test1', help='测试名称')
    parser.add_argument('--test_midi_file', type=str, default='test1', help='测试 MIDI 文件路径')
    
    # # 解析命令行参数
    args = parser.parse_args()
    ds_to_midi(args.test_name, args.test_midi_file)
