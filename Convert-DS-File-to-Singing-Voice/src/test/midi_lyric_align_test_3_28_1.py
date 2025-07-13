import pretty_midi
import librosa
import os
import cn2an
import argparse

from src.lyric_to_phoneme import _get_char_type
import src.lyric_to_phoneme as l2p
import src.midi_lyric_align_helper_func as helper
import src.read_file as rf
from src.params import song_ids

AP_time = 0.2
AP_min_time = 0.1
SP_min_time = 0.1

# 读取列表中每个字典的信息，并生成音符序列
def generate_notes(data, midi_data, pinyin_phoneme_dict, bpm=60):
    min_note_dur = 60 / bpm / 8 # or 60 / bpm / 4
    singing_longest = 2
    singing_longest_last = singing_longest + min_note_dur
    word_note_data = []
    
    last_d_i = len(data) - 1
    for d_i , record in enumerate(data):
        note_seq = []
        note_dur = []
        note_slur = []
        word_dur_new = []
        word_seq_new = []
        note_offset = -1

        # 提取音符信息
        word_seq, word_dur, word_start, word_end = helper.words_seperate(record['words'])

        if word_seq == ['']: # 去除空句子
            continue
        for i, word in enumerate(word_seq):# 去除标识多人唱歌手的符号之前的内容，如：'男:', '孩子:'
            if word == ":" or word == "：":
                while i >= 0:
                    word_seq[i] = 'SP'
                    i -= 1
        
        last_w_i = len(word_seq) - 1
        for w_i, (word, dur, start_time, end_time) in enumerate(zip(word_seq, word_dur, word_start, word_end)):
            if w_i == last_w_i and d_i != last_d_i:
                if end_time > float(data[d_i + 1]['start'])/1000:
                    end_time = float(data[d_i + 1]['start'])/1000
            assert start_time <= end_time, f"start_time {start_time} >= end_time {end_time}"

            note_avail = []

            # 非中文：阿拉伯数字转换为相应的中文数字，然后英文先变为SP、rest
            word_type = _get_char_type(word[0])
            if word_type != 'chinese':
                if word_type == 'en_digit':
                    word = cn2an.an2cn(word, "low")
                elif note_offset > 0:
                    word_seq_new.append(word)
                    word_dur_new.append(dur)
                    note_seq.append('rest')
                    note_dur.append(dur)
                    note_slur.append(0)
                    continue
            
            # 将时间范围内的音符放入候选名单，另外，为了防止时间戳太短导致音符数量不够，还把其下一个音符暂时纳入
            for instrument in midi_data.instruments: 
                if instrument.is_drum or instrument.program != 0:
                    continue
                for note in instrument.notes:
                    if (note.start >= start_time and note.start <= end_time) or (note.end >= start_time + min_note_dur/2 and note.end <= end_time) or (note.start <= start_time and note.end >= end_time) or note.end == instrument.notes[-1].end:
                        note_avail.append(note)
                    if note.start > end_time:
                        note_avail.append(note)
                        break
            
            # 排序 按照音符的开始时间离start_time的距离
            note_sorted = sorted(note_avail, key=lambda x: abs(x.start - start_time) + abs(x.end - end_time))
            # note_sorted = sorted(note_avail, key=lambda x: helper.get_note_word_ratio(x.start, x.end, start_time, end_time))
            note_closest = note_sorted[0]
            
            slur = 0
            note_time_sum = 0
            note_time_sum += note_closest.end - note_closest.start

            # threshold 为音符时长已满足字时长的门槛
            threshold = min_note_dur
            
            # 开始处理音符
            if note_time_sum >= threshold: # 最接近的 note 约等于 word duration 直接加入
                # 3-27 test1：最简单的一字一音，每个字取原时间戳          
                final_dur = dur # if w_i != last_w_i else note_time_sum # 最后一个字的音符时长取最长, 其他的取字的时长
                
                # 处理中文数字为多个的情况
                if len(word) > 1:
                    for char in word:
                        word_seq_new.append(char)
                        word_dur_new.append(final_dur/len(word))
                else:
                    word_seq_new.append(word)
                    word_dur_new.append(final_dur)
                
                note_dur.append(final_dur)
                note_name = librosa.midi_to_note(note_closest.pitch)
                note_seq.append(note_name)
                note_slur.append(slur)
                if note_offset < 0:
                    note_offset = note_closest.start
            else:
                raise ValueError(f'note_time_sum {note_time_sum} < threshold {threshold}')
            
            # 两个字之间有空隙，补充AP SP rest
            if w_i != last_w_i and end_time != word_end[-1] and end_time != word_start[w_i + 1]: 
                gap = word_start[w_i + 1] - start_time - dur
                if gap > 0: # 有空隙
                    if gap > AP_time: # 有空隙，且大于AP_time
                        SP_time = gap - AP_time
                        if SP_time >= SP_min_time: # SP_time 大于 SP_min_time 才需要补充SP
                            note_seq.append('rest')
                            note_dur.append(SP_time)
                            note_slur.append(0)
                            word_seq_new.append('SP')
                            word_dur_new.append(SP_time)
                        else: # SP_time 小于 SP_min_time，加到上一个音符上
                            note_dur[-1] += SP_time
                            word_dur_new[-1] += SP_time
                        
                        note_seq.append('rest')
                        note_dur.append(AP_time)
                        note_slur.append(0)
                        word_seq_new.append('AP')
                        word_dur_new.append(AP_time)
                    elif gap >= AP_min_time: # 有空隙，且大于AP_min_time
                        note_seq.append('rest')
                        note_dur.append(gap)
                        note_slur.append(0)
                        word_seq_new.append('AP')
                        word_dur_new.append(gap)
                    else:  # 有空隙，但小于AP_min_time，加到上一个音符上
                        note_dur[-1] += gap
                        word_dur_new[-1] += gap
        
        ph_seq = l2p.word_to_phoneme(word_seq_new, pinyin_phoneme_dict)
        if ph_seq == '':
            continue

        word_note = {}
        # word_note['offset'] = float(record['start'])/1000
        origin_offset = float(record['start'])/1000
        word_note['offset'] = note_offset if abs(origin_offset - note_offset) < 2 * min_note_dur else origin_offset
        word_note['word_seq'] = ' '.join(word_seq_new)
        word_note['word_dur'] = ' '.join(map(str, word_dur_new))
        word_note['ph_seq'] = ph_seq
        word_note['note_seq'] = ' '.join(note_seq)
        word_note['note_dur'] = ' '.join(map(str, note_dur))
        word_note['note_slur'] = ' '.join(map(str, note_slur))
        
        word_note_data.append(word_note)
        
    return word_note_data
        
def merge_records(record1, record2, gap):
    if gap > AP_time:
        merged_word_seq = record1['word_seq'] + ' SP AP ' + record2['word_seq']  
        merged_word_dur = record1['word_dur'] + f' {gap - AP_time} {AP_time} ' + record2['word_dur']  
        merged_offset = record1['offset']  # 使用第一个记录的offset作为合并后的offset 
        merged_ph_seq = record1['ph_seq'] + ' SP AP ' + record2['ph_seq']
        merged_note_seq = record1['note_seq'] + ' rest rest ' + record2['note_seq']
        merged_note_dur = record1['note_dur'] + f' {gap - AP_time} {AP_time} ' + record2['note_dur']
        merged_note_slur = record1['note_slur'] + ' 0 0 ' + record2['note_slur']
    elif gap > AP_min_time:
        merged_word_seq = record1['word_seq'] + ' AP ' + record2['word_seq']  
        merged_word_dur = record1['word_dur'] + f' {gap} ' + record2['word_dur']  
        merged_offset = record1['offset']
        merged_ph_seq = record1['ph_seq'] + ' AP ' + record2['ph_seq']
        merged_note_seq = record1['note_seq'] + ' rest ' + record2['note_seq']
        merged_note_dur = record1['note_dur'] + f' {gap} ' + record2['note_dur']
        merged_note_slur = record1['note_slur'] + ' 0 ' + record2['note_slur']
    elif gap > 0:
        merged_word_seq = record1['word_seq'] + ' ' + record2['word_seq']
        record1_word_dur = helper.word_dur_to_seconds(record1['word_dur'])
        record1_word_dur[-1] += gap
        record1['word_dur'] = ' '.join(map(str, record1_word_dur))
        merged_word_dur = record1['word_dur'] + ' ' + record2['word_dur']  
        merged_offset = record1['offset']
        merged_ph_seq = record1['ph_seq'] + ' ' + record2['ph_seq']
        merged_note_seq = record1['note_seq'] + ' ' + record2['note_seq']
        record1_note_dur = helper.word_dur_to_seconds(record1['note_dur'])
        record1_note_dur[-1] += gap
        record1['note_dur'] = ' '.join(map(str, record1_note_dur))
        merged_note_dur = record1['note_dur'] + ' ' + record2['note_dur']
        merged_note_slur = record1['note_slur'] + ' ' + record2['note_slur']
    else:
        merged_word_seq = record1['word_seq'] + ' ' + record2['word_seq']
        record1_word_dur = helper.word_dur_to_seconds(record1['word_dur'])
        record1_word_dur[-1] += gap
        assert record1_word_dur[-1] > 0, f"record1_word_dur[-1] {record1_word_dur[-1]} <= 0"
        record1['word_dur'] = ' '.join(map(str, record1_word_dur))
        merged_word_dur = record1['word_dur'] + ' ' + record2['word_dur']  
        merged_offset = record1['offset']
        merged_ph_seq = record1['ph_seq'] + ' ' + record2['ph_seq']
        merged_note_seq = record1['note_seq'] + ' ' + record2['note_seq']
        record1_note_dur = helper.word_dur_to_seconds(record1['note_dur'])
        record1_note_dur[-1] += gap
        assert record1_note_dur[-1] > 0, f"record1_note_dur[-1] {record1_note_dur[-1]} <= 0"
        record1['note_dur'] = ' '.join(map(str, record1_note_dur))
        merged_note_dur = record1['note_dur'] + ' ' + record2['note_dur']
        merged_note_slur = record1['note_slur'] + ' ' + record2['note_slur']
    return {  
        'offset': merged_offset,  
        'word_seq': merged_word_seq,  
        'word_dur': merged_word_dur,
        'ph_seq': merged_ph_seq,
        'note_seq': merged_note_seq,
        'note_dur': merged_note_dur,
        'note_slur': merged_note_slur
    }

def check_offsets(songID, data):
    if data[0]['offset'] > AP_time:  
        data[0] = helper.add_AP_front(data[0], AP_time)
    i = 0  
    min_interval = AP_min_time
    while i < len(data):
        merged = False
        while i + 1 < len(data):  
            current_record = data[i]  
            next_record = data[i + 1]  
            
            end_time = current_record['offset'] + sum(helper.word_dur_to_seconds(current_record['word_dur'])) 

            if end_time > next_record['offset']:    
                print(f"{songID} offset {current_record['offset']} end_time {end_time} next_record {next_record['offset']} dis {end_time - next_record['offset']}") 
            if end_time > next_record['offset'] - min_interval:  
                # 合并当前记录和下一个记录  
                merged_record = merge_records(current_record, next_record, next_record['offset'] - end_time)  
                
                # 更新列表中的记录  
                data[i] = merged_record  
                del data[i + 1]  
                
                # 标记为已合并  
                merged = True  
                break  # 跳出内部循环，重新检查合并后的记录与下一个记录的关系  
            else:
                try:
                    if data[i + 1]['word_seq'][0] !='S' and data[i + 1]['word_seq'][0] !='A' :
                        data[i + 1] = helper.add_AP_front(next_record)
                except:
                    raise ValueError(f'index {i + 1} {data[i + 1]}')
                # 不需要合并，退出内部循环  
                break  
        
        # 如果没有合并，或者已经合并并更新了列表，移动到下一个记录  
        if not merged:  
            i += 1  
    
    return data 



def main(file_name, test_name, test_midi_file):
    bpm_midi_file = f'data/bpm_midi/{test_name}/{file_name}_src.mp3_5b.mid'
    midi_file = f'data/midi/{test_midi_file}/{file_name}_new.mid'
    json_file = f'data/json/{test_name}/{file_name}.json'
    ds_file = f'data/ds/{test_midi_file}/{file_name}.ds'
    dict_file='DiffSinger/dictionaries/opencpop-extension.txt'
    
    pinyin_phoneme_dict = rf.read_dict(dict_file)
    bpm = rf.read_midi_get_bpm(bpm_midi_file)
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    json_data = rf.read_changba_json(json_file)
    word_note_data = generate_notes(json_data, midi_data, pinyin_phoneme_dict, bpm)
    merged_word_note_data = check_offsets(file_name, word_note_data)

    # 将 JSON 数据列表写入文件
    os.makedirs(os.path.dirname(ds_file), exist_ok=True)
    helper.add_ph_num(merged_word_note_data, ds_file, dictionary=dict_file)

def midi_lyric_align(test_name, test_midi_file):
    for song_id in song_ids[test_name]:
        try:
            main(str(song_id), test_name, test_midi_file)
        except:
            raise ValueError(f'Error occured when processing song {song_id}')
    
if __name__ == '__main__':
    # 创建解析器
    parser = argparse.ArgumentParser()
    
    # 添加参数
    parser.add_argument('--test_name', type=str, default='test1', help='测试名称')
    parser.add_argument('--test_midi_file', type=str, default='test1', help='测试 MIDI 文件路径')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用函数并传递参数
    midi_lyric_align(args.test_name, args.test_midi_file)
