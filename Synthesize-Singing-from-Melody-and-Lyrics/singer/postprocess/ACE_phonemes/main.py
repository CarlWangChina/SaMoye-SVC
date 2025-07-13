
# 自动获取本文件路径，添加本文件路径到sys.path
import os
import os.path as osp
import sys
sys.path.append(osp.dirname(osp.abspath(__file__)))

import src.cmudict_reader as cmudict_reader
import src.plan_reader as plan_reader
import re


def save_all_phoneme_to_md(path="phonemes.md"):
    with open(path, "w") as f:
        f.write("# Phonemes\n")
        f.write("| 语言 Language | 辅音 Heads | 元音 Tails |\n| --- | --- | --- |\n")
        for plan in [plan_reader.zh_plan, plan_reader.jp_plan, plan_reader.en_plan]:
            f.write("| {} | {} | {} |\n".format(plan["language"], plan["phon_class"]["head"], plan["phon_class"]["tail"]))
        f.write("\n\n")


# phoneme validation 
# language：zh jp eng
def is_valid_phoneme(phoneme, language):

    assert language in ["zh", "jp", "eng"]

    all_phonemes = []
    for plan in [plan_reader.zh_plan, plan_reader.jp_plan, plan_reader.en_plan]:
        if plan["language"] == language:
            all_phonemes.extend(plan["phon_class"]["head"])
            all_phonemes.extend(plan["phon_class"]["tail"])
            break

    return phoneme in all_phonemes



# 拼音到音素
def pinyin_to_phoneme(pinyin):
    zh_plan = plan_reader.zh_plan
    
    if pinyin in zh_plan["dict"]:
        return zh_plan["dict"][pinyin]
    elif pinyin in zh_plan["syllable_alias"]:
        return zh_plan["dict"][zh_plan["syllable_alias"][pinyin]]
    else:
        return "pinyin not found"



# 日本語変換音素
def jp_word_to_phoneme(jp_word):
    
    jp_plan = plan_reader.jp_plan
    jp_word2romaji = plan_reader.jp_word2romaji
    
    if jp_word in jp_word2romaji:
        jp_word = jp_word2romaji[jp_word]
    
    if jp_word in jp_plan["dict"]:
        return jp_plan["dict"][jp_word]
    elif jp_word in jp_plan["syllable_alias"]:
        return jp_plan["dict"][jp_plan["syllable_alias"][jp_word]]
    else:
        return "word not found"


def find_all_patterns(lst, target):
    indices = []
    n = len(target)
    i = 0
    while i <= len(lst) - n:
        lst_slice = lst[i:i + n]
        compare_slice = []
        for phn in lst_slice:
            if phn in plan_reader.en_plan["phon_class"]["tail"]:
                compare_slice.append("vowel")
            else:
                compare_slice.append(phn)
                
        if compare_slice == target:
            indices.append(i)
            i += n  # Move i forward by n steps
        else:
            i += 1  # Move i forward by 1 step
            
    return indices


def replace_elements(arr, start_idx, num_elements, sub_arr):
    return arr[:start_idx] + sub_arr + arr[start_idx + num_elements:]


def find_and_replace_all_patterns(lst, target, replacement, replace_partial=False):
    indices = find_all_patterns(lst, target)
    for idx in indices:
        if replace_partial:
            lst = replace_elements(lst, idx, len(replacement), replacement)
        else:
            lst = replace_elements(lst, idx, len(target), replacement)
    return lst


# t r -> tr
# d r -> dr

# s t vowel --> s d vowel
# s k vowel --> s g vowel
# s p vowel --> s b vowel
# s tr vowel --> s dr vowel
def eng_phoneme_normalize(syllable):
    phonemes = []
    for i in range(len(syllable)):
        phn = syllable[i].lower()
        if re.search(r'\d$', phn):
            phn = phn[:-1]
        phonemes.append(phn)

    # phonemes = find_and_replace_all_patterns(phonemes, ['t', 'r'], ['tr'])
    # phonemes = find_and_replace_all_patterns(phonemes, ['d', 'r'], ['dr'])
    
    # phonemes = find_and_replace_all_patterns(phonemes, ['s', 't', 'vowel'], ['s', 'd'], True)
    # phonemes = find_and_replace_all_patterns(phonemes, ['s', 'k', 'vowel'], ['s', 'g'], True)
    # phonemes = find_and_replace_all_patterns(phonemes, ['s', 'p', 'vowel'], ['s', 'b'], True)
    # phonemes = find_and_replace_all_patterns(phonemes, ['s', 'tr', 'vowel'], ['s', 'dr'], True)

    return phonemes


# english word to syllable and phoneme
def eng_word_to_phoneme(en_word):

    eng_dict = cmudict_reader.get_dict()

    word_key = en_word.upper()

    if word_key in eng_dict:
        syllables = eng_dict[word_key]

        syllables_normalized = []
        for phn_list in syllables:
            phonemes = eng_phoneme_normalize(phn_list)
            syllables_normalized.append(phonemes)
            

        return syllables_normalized
    else:
        return "word not found"


if __name__ == "__main__":
    save_all_phoneme_to_md()

    print(is_valid_phoneme("ah", "eng"))

    # pinyin_to_phoneme
    print("==========================")
    print(pinyin_to_phoneme("pin"))
    print(pinyin_to_phoneme("lve"))
    print(pinyin_to_phoneme("lue"))
    print(pinyin_to_phoneme("asd"))

    # jp_word_to_phoneme
    print("==========================")
    print(jp_word_to_phoneme("ヴぁ"))
    print(jp_word_to_phoneme("ja"))
    print(jp_word_to_phoneme("jya"))
    print(jp_word_to_phoneme("asd"))

    # eng_word_to_phoneme
    print("==========================")
    print(eng_word_to_phoneme("yesterday"))
    print(eng_word_to_phoneme("untrue"))
    print(eng_word_to_phoneme("arrested"))
    print(eng_word_to_phoneme("favorite"))