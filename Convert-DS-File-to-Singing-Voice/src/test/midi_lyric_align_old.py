import pretty_midi
import librosa
import os
import cn2an
from src.lyric_to_phoneme import _get_char_type
import src.lyric_to_phoneme as l2p
import src.midi_lyric_align_helper_func as helper
import src.read_file as rf
import argparse
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

        # 提取音符信息
        word_seq, word_dur, word_start, word_end = helper.words_seperate(record['words'])
        # print(word_seq, word_dur)
        if word_seq == ['']: # 空句子
            continue
        for i, word in enumerate(word_seq):
            if word == ":" or word == "：":
                while i >= 0:
                    word_seq[i] = 'SP'
                    i -= 1
        
        last_w_i = len(word_seq) - 1
        for w_i, (word, dur, start_time, end_time) in enumerate(zip(word_seq, word_dur, word_start, word_end)):
            note_avail = []

            word_type = _get_char_type(word[0])
            if word_type != 'chinese':
                if word_type == 'en_digit':
                    word = cn2an.an2cn(word, "low")
                else:
                    word_seq_new.append(word)
                    word_dur_new.append(dur)
                    note_seq.append('rest')
                    note_dur.append(dur)
                    note_slur.append(0)
                    continue
            
            for instrument in midi_data.instruments: # 时间范围内的音符
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
            note_closest = note_sorted[0]
            
            slur = 0
            note_time_sum = 0
            note_time_sum += note_closest.end - note_closest.start

            # threshold 为音符时长已满足字时长的门槛
            threshold = dur - 2 * min_note_dur if dur - 2 * min_note_dur > min_note_dur else min_note_dur
            if w_i == last_w_i:# 最后一个字的门槛高点
                threshold = dur + 1 * min_note_dur

            # 最长音符时长：为了保证每个字的音符时长不超过字的时长太多
            longest_dur = dur if dur > min_note_dur else min_note_dur
            if w_i == last_w_i and longest_dur <= 0.8:# 每句话的最后一个字，最长取剩余时间，让每句话最后一个字的长音效果正常
                if d_i < last_d_i: # 不是最后一句
                    rest_dur = float(data[d_i+1]['start'])/1000 - (float(record['start'])/1000 + sum(word_dur_new))
                    # print(data[d_i+1]['start'], record['start'], word_dur_new, sum(word_dur_new), rest_dur)
                else: # 最后一句
                    rest_dur = float(record['end'])/1000 - (float(record['start'])/1000 + sum(word_dur_new))
                    if rest_dur < 8 * min_note_dur:
                        rest_dur = 8 * min_note_dur
                # 做点限制，防止超过下一句
                if rest_dur > 0:
                    if rest_dur - AP_min_time < 2 * min_note_dur :
                        longest_dur = rest_dur - min_note_dur if rest_dur - min_note_dur > min_note_dur else rest_dur
                    else: 
                        longest_dur = rest_dur - AP_min_time if rest_dur - AP_min_time > min_note_dur else rest_dur
            assert longest_dur > 0, f'word {word} longest_dur {longest_dur} \
            rest_dur {rest_dur} word_dur_new {word_dur_new} word_dur {word_dur} record {record}\
                slur {slur} note_time_sum {note_time_sum} dur {dur} threshold {threshold} w_i {w_i} last_w_i {last_w_i} d_i {d_i} last_d_i {last_d_i}'
            
            # 开始处理音符
            if note_time_sum >= threshold: # 最接近的 note 约等于 word duration 直接加入
                # 没有转音符
                if note_time_sum > longest_dur :
                    note_time_sum = longest_dur
                
                final_dur = dur if w_i != last_w_i else note_time_sum # 最后一个字的音符时长取最长, 其他的取字的时长
                
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

                if final_dur > singing_longest: # 限制唱的最长的时间，超过的部分用AP补充
                    if w_i == last_w_i: # 最后一个字若长于singing_longest_last则替换为singing_longest_last，否则不需要变
                        if final_dur > singing_longest_last: # 最后一个不需要补充AP
                            note_dur[-1] = singing_longest_last
                            word_dur_new[-1] = singing_longest_last
                    else: 
                        over_time = final_dur - singing_longest
                        if over_time >= AP_min_time: # 超出的时间大于AP_min_time才需要补充AP
                            note_dur[-1] = singing_longest
                            word_dur_new[-1] = singing_longest
                            # 非最后一个要补充AP
                            word_seq_new.append('AP')
                            word_dur_new.append(over_time)
                            note_seq.append('rest')
                            note_dur.append(over_time)
                            note_slur.append(0)

                # print(f'{slur} {word} {note_name} dur: {final_dur}')
                # print(f'{word} word_dur: {final_dur} dur: {dur} dis {final_dur - dur}')
            else:
                slur_notes = []
                slur_notes.append(note_closest)
                # 将最接近的音符们放到候选名单中
                for note in note_sorted:
                    if note.start != note_closest.start:
                        note_duration = note.end - note.start
                        note_time_sum += note_duration
                        slur_notes.append(note)
                    if note_time_sum >= threshold:
                        break
                
                # 按照音符的开始时间排序，然后依次加入音符
                slur_notes = sorted(slur_notes, key=lambda x: x.start)
                note_time_sum = 0
                for note in slur_notes:
                    note_duration = note.end - note.start
                    note_time_sum += note_duration
                    note_name = librosa.midi_to_note(note.pitch)
                    if slur == 0 or note_name != note_seq[-1]:
                        note_seq.append(note_name)
                        note_slur.append(slur)
                        note_dur.append(note_duration)
                    else:
                        note_dur[-1] = note_dur[-1] + note_duration
                    slur = 1

                    # 超过最长音符时长，不再加入音符
                    if note_time_sum > longest_dur or (note_time_sum > singing_longest and w_i != last_w_i) or note_time_sum > singing_longest_last:
                        break
                    
                if note_time_sum < dur : # 防止由于音符少，导致最后的音符比word duration短导致不好听
                    note_dur[-1] = note_dur[-1] + dur - note_time_sum
                    note_time_sum = dur

                rest_dur = 0
                if note_time_sum > singing_longest_last and w_i == last_w_i: # 限制唱的最长的时间，超过的部分用AP补充
                    note_dur[-1] -= note_time_sum - singing_longest_last # 最后一个不需要补充AP
                    note_time_sum = singing_longest_last
                elif note_time_sum > singing_longest and w_i != last_w_i: # 非最后一个要补充AP
                    rest_dur = note_time_sum - singing_longest
                    note_dur[-1] -= rest_dur
                    note_time_sum = singing_longest
                elif note_time_sum > longest_dur: # 大于 longest_dur 不需要补AP
                    note_dur[-1] -= note_time_sum - longest_dur
                    note_time_sum = longest_dur
                assert note_dur[-1] > 0, f'word {word} note_dur {note_dur} note_time_sum {note_time_sum} dur {dur} longest_dur {longest_dur} singing_longest {singing_longest} singing_longest_last {singing_longest_last} word {word}'
                
                if len(word) > 1:
                    for char in word:
                        word_seq_new.append(char)
                        word_dur_new.append(note_time_sum/len(word))
                else:
                    word_seq_new.append(word)
                    word_dur_new.append(note_time_sum)
                if rest_dur >= AP_min_time: # 超出的时间大于AP_min_time才需要补充AP
                    word_seq_new.append('AP')
                    word_dur_new.append(rest_dur)
                    note_seq.append('rest')
                    note_dur.append(rest_dur)
                    note_slur.append(0)
                # print(f'{word} word_dur: {note_time_sum} dur: {dur} dis {note_time_sum - dur}')
            
            if end_time != word_end[-1] and end_time != word_start[w_i + 1]: # 两个字之间有空隙，补充AP SP rest
                gap = word_start[w_i + 1] - start_time - note_time_sum
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
        word_note['ph_seq'] = ph_seq
        word_note['offset'] = float(record['start'])/1000
        word_note['word_seq'] = ' '.join(word_seq_new)
        word_note['word_dur'] = ' '.join(map(str, word_dur_new))
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
        merged_word_dur = record1['word_dur'] + ' ' + record2['word_dur']  
        merged_offset = record1['offset']
        merged_ph_seq = record1['ph_seq'] + ' ' + record2['ph_seq']
        merged_note_seq = record1['note_seq'] + ' ' + record2['note_seq']
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

def check_offsets(data):
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
                print(f"offset {current_record['offset']} end_time {end_time} next_record {next_record['offset']} dis {end_time - next_record['offset']}") 
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
    merged_word_note_data = check_offsets(word_note_data)

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