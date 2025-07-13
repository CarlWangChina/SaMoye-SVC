import json
import pathlib


def word_dur_to_seconds(word_dur):
    return [float(dur) for dur in word_dur.split()]


def words_seperate(words):
    word_seq = []
    word_dur = []
    word_start = []
    word_end = []
    for word in words:
        word_seq.append(word[2])
        word_dur.append(float(word[1] - word[0]) / 1000)
        word_start.append(float(word[0]) / 1000)
        word_end.append(float(word[1]) / 1000)
    return word_seq, word_dur, word_start, word_end


def add_AP_front(record, AP_time=0.1):
    record["offset"] = record["offset"] - AP_time
    record["word_seq"] = "AP " + record["word_seq"]
    record["word_dur"] = f"{AP_time} " + record["word_dur"]
    record["ph_seq"] = "AP " + record["ph_seq"]
    record["note_seq"] = "rest " + record["note_seq"]
    record["note_dur"] = f"{AP_time} " + record["note_dur"]
    record["note_slur"] = "0 " + record["note_slur"]
    return record


def add_ph_num(
    items,
    transcription: str,
    dictionary: str = None,
    vowels: str = None,
    consonants: str = None,
):
    assert dictionary is not None or (
        vowels is not None and consonants is not None
    ), "Either dictionary file or vowels and consonants file should be specified."
    if dictionary is not None:
        dictionary = pathlib.Path(dictionary).resolve()
        vowels = {"SP", "AP"}
        consonants = set()
        with open(dictionary, "r", encoding="utf8") as f:
            rules = f.readlines()
        for r in rules:
            syllable, phonemes = r.split("\t")
            phonemes = phonemes.split()
            assert (
                len(phonemes) <= 2
            ), "We only support two-phase dictionaries for automatically adding ph_num."
            if len(phonemes) == 1:
                vowels.add(phonemes[0])
            else:
                consonants.add(phonemes[0])
                vowels.add(phonemes[1])
    else:
        vowels_path = pathlib.Path(vowels).resolve()
        consonants_path = pathlib.Path(consonants).resolve()
        vowels = {"SP", "AP"}
        consonants = set()
        with open(vowels_path, "r", encoding="utf8") as f:
            vowels.update(f.read().split())
        with open(consonants_path, "r", encoding="utf8") as f:
            consonants.update(f.read().split())
        overlapped = vowels.intersection(consonants)
        assert len(vowels.intersection(consonants)) == 0, (
            "Vowel set and consonant set overlapped. The following phonemes "
            "appear both as vowels and as consonants:\n"
            f"{sorted(overlapped)}"
        )

    for item in items:
        item: dict
        ph_seq = item["ph_seq"].split()
        for ph in ph_seq:
            assert (
                ph in vowels or ph in consonants
            ), f'Invalid phoneme symbol \'{ph}\' in \'{item["name"]}\'.'
        ph_num = []
        i = 0
        while i < len(ph_seq):
            j = i + 1
            while j < len(ph_seq) and ph_seq[j] in consonants:
                j += 1
            ph_num.append(str(j - i))
            i = j
        item["ph_num"] = " ".join(ph_num)

    # print(items)
    with open(transcription, "w", encoding="utf8") as f:
        json.dump(items, f, ensure_ascii=False, indent=4)


def mq_add_ph_num(
    items, dictionary: str = None, vowels: str = None, consonants: str = None
):
    assert dictionary is not None or (
        vowels is not None and consonants is not None
    ), "Either dictionary file or vowels and consonants file should be specified."
    if dictionary is not None:
        dictionary = pathlib.Path(dictionary).resolve()
        vowels = {"SP", "AP"}
        consonants = set()
        with open(dictionary, "r", encoding="utf8") as f:
            rules = f.readlines()
        for r in rules:
            syllable, phonemes = r.split("\t")
            phonemes = phonemes.split()
            assert (
                len(phonemes) <= 2
            ), "We only support two-phase dictionaries for automatically adding ph_num."
            if len(phonemes) == 1:
                vowels.add(phonemes[0])
            else:
                consonants.add(phonemes[0])
                vowels.add(phonemes[1])
    else:
        vowels_path = pathlib.Path(vowels).resolve()
        consonants_path = pathlib.Path(consonants).resolve()
        vowels = {"SP", "AP"}
        consonants = set()
        with open(vowels_path, "r", encoding="utf8") as f:
            vowels.update(f.read().split())
        with open(consonants_path, "r", encoding="utf8") as f:
            consonants.update(f.read().split())
        overlapped = vowels.intersection(consonants)
        assert len(vowels.intersection(consonants)) == 0, (
            "Vowel set and consonant set overlapped. The following phonemes "
            "appear both as vowels and as consonants:\n"
            f"{sorted(overlapped)}"
        )

    for item in items:
        item: dict
        print(item)
        ph_seq = item["ph_seq"].split()
        for ph in ph_seq:
            assert (
                ph in vowels or ph in consonants
            ), f'Invalid phoneme symbol \'{ph}\' in \'{item["name"]}\'.'
        ph_num = []
        i = 0
        while i < len(ph_seq):
            j = i + 1
            while j < len(ph_seq) and ph_seq[j] in consonants:
                j += 1
            ph_num.append(str(j - i))
            i = j
        item["ph_num"] = " ".join(ph_num)

    # print(items)
    return items


def get_note_word_ratio(note_start, note_end, word_start, word_end):
    if word_start == word_end:
        return 0
    if note_start > word_end or note_end < word_start:
        return 0
    if note_start >= word_start and note_end <= word_end:  # note在word中
        return abs((note_end - note_start) / (word_end - word_start))
    elif note_start <= word_start and note_end >= word_end:  # word在note中
        return 1
    elif note_start < word_start and note_end <= word_end:  # note在word前部分
        return abs((note_end - word_start) / (word_end - word_start))
    elif note_start >= word_start and note_end > word_end:  # note在word后部分
        return abs((word_end - note_start) / (word_end - word_start))
    else:  # note和word没有交集
        return 0
