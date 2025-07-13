from src.params import song_ids
import re
import src.read_file as rf
from src.lyric_to_phoneme import _get_char_type
import os
import json

lyrics = {"一二三四五六七八九十"}
lyrics = list(lyrics.pop())

def _change_lyric(song_id, test_name):
    json_file = f'data/json/{test_name}/{song_id}.json'
    new_json_file = f'data/json/{test_name}/{song_id}_modified.json'
    new_lyric_file = f'data/lrc_modified/{test_name}/{song_id}_modified.txt'

    if os.path.exists(new_lyric_file):
        with open(new_lyric_file, 'r') as f:
            new_lyric = f.read()
            # 去除特殊符号后，按字分割成列表
            new_lyric_lists = [re.findall(r"[\u4e00-\u9fff]|[a-zA-Z]+", line) for line in new_lyric.split("\n")]
    else:
        new_lyric_lists = [lyrics]
    # print(new_lyric_list)

    json_data = rf.read_changba_json(json_file)
    new_lyric_lists_len = len(new_lyric_lists)
    
    for i, record in enumerate(json_data):
        words = record['words']
        new_lyric_list = new_lyric_lists[i%new_lyric_lists_len]
        new_lyric_list_len = len(new_lyric_list)
        word_count = 0
        for j, word in enumerate(words):
            word_type = _get_char_type(word[2])
            if word_type == 'chinese' or word_type == 'english':
                words[j][2] = new_lyric_list[word_count % new_lyric_list_len]
                word_count += 1
        record['text'] = ''.join([word[2] for word in words])
    with open(new_json_file, 'w', encoding='utf8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
    print(f"Lyric changed for {song_id} in {test_name}")

def change_lyric(test_input):
    for song_id in song_ids[test_input.test_name]:
        try:
            _change_lyric(song_id, test_input.test_name)
        except Exception as e:
            raise ValueError(f"Error in changing lyric for {song_id} in {test_input.test_name}: {e}")

def _change_not_cn_lyric(song_id, test_name):
    json_file = f'data/json/{test_name}/{song_id}.json'
    new_json_file = f'data/json/{test_name}/{song_id}_fixed.json'
    new_lyric_lists = [lyrics]

    json_data = rf.read_changba_json(json_file)
    new_lyric_lists_len = len(new_lyric_lists)
    
    for i, record in enumerate(json_data):
        words = record['words']
        new_lyric_list = new_lyric_lists[i%new_lyric_lists_len]
        new_lyric_list_len = len(new_lyric_list)
        word_count = 0
        for j, word in enumerate(words):
            word_type = _get_char_type(word[2])
            if word_type != 'chinese' and word_type != 'en_digit' and word_type != "space" and word_type != "sep":
                print(word_type, words[j][2])
                words[j][2] = new_lyric_list[word_count % new_lyric_list_len]
                word_count += 1
        record['text'] = ''.join([word[2] for word in words])
    with open(new_json_file, 'w', encoding='utf8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
    # print(f"Lyric changed for {song_id} in {test_name}")

def change_not_cn_lyric(test_input):
    for song_id in song_ids[test_input.test_name]:
        try:
            _change_not_cn_lyric(song_id, test_input.test_name)
        except Exception as e:
            raise ValueError(f"Error in changing lyric for {song_id} in {test_input.test_name}: {e}")

if __name__ == '__main__':
    class TestInput:
        def __init__(self, test_name):
            self.test_name = test_name
    test_input = TestInput('singer600_2')
    # change_lyric(test_input)
    change_not_cn_lyric(test_input)