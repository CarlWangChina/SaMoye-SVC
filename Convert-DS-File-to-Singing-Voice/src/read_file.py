import json
import pretty_midi

# 读取jsonl文件并解析为字典列表
def read_changba_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            data.append(record)
    return data

# 读取jsonl文件并解析为字典列表
def read_changba_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# 读取字典文件，生成字典
def read_dict(dict_file='/export/data/home/john/MuerSinger2/DiffSinger/dictionaries/opencpop-extension.txt'):
    pinyin_phoneme_dict = {}
    with open(dict_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip().split('\t')
            pinyin = line[0]
            phoneme = line[1]
            pinyin_phoneme_dict[pinyin] = phoneme
    return pinyin_phoneme_dict

def read_midi_get_bpm(bpm_midi_file):
    bpm_midi_data = pretty_midi.PrettyMIDI(bpm_midi_file)
    bpms = []
    for instrument in bpm_midi_data.instruments:
        if instrument.is_drum:
            for i, note in enumerate(instrument.notes):
                if i == 0:
                    bpms.append(round(note.end - note.start, 3))
                    continue
                note_time = round(note.start - instrument.notes[i-1].start, 3)
                if note_time not in bpms:
                    bpm = note_time
                    bpms.append(bpm)
    bpm = 60 / (sum(bpms) / len(bpms))
    # print(bpm,  60 / bpm / 8 , bpms)
    return bpm