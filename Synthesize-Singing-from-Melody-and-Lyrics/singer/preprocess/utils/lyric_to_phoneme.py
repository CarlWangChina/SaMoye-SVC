from pypinyin import lazy_pinyin

#######################################################################################################
# Constants
#######################################################################################################
PINYIN_TO_PHONEME_DICT_FILE = "pinyin2phoneme.txt"
SPECIAL_TOKEN_PATTERN = r"<\|.*?\|>"
SEP_CHARACTERS = [
    "，",
    "。",
    "？",
    "：",
    "！",
    "；",
    ",",
    ".",
    ":",
    ";",
    "?",
    "!",
    "、",
    "/",
    "\\",
    "\r",
    "\n",
]
SPACE_CHARACTERS = [" ", "　", "\t"]

#######################################################################################################
# Character type ranges (based on Unicode)
# arranged in (start, end, type)
#######################################################################################################
CHAR_TYPE_RANGE_MAP = [
    (0x27, 0x27, "en_alphabet"),  # '
    (0x30, 0x39, "en_digit"),  # 0-9
    (0x41, 0x5A, "en_alphabet"),  # A-Z
    (0x61, 0x7A, "en_alphabet"),  # a-z
    (0x100, 0x17F, "latin_alphabet"),  # Latin Extended-A
    (0x180, 0x24F, "latin_alphabet"),  # Latin Extended-B
    (0x370, 0x3FF, "greek"),  # Greek and Coptic
    (0x400, 0x4FF, "cyrillic"),  # Cyrillic
    (0x530, 0x58F, "armenian"),  # Armenian
    (0x590, 0x5FF, "hebrew"),  # Hebrew
    (0x600, 0x6FF, "arabic"),  # Arabic
    (0x700, 0x74F, "syriac"),  # Syriac
    (0x750, 0x77F, "arabic"),  # Arabic Supplement
    (0x780, 0x7BF, "thaana"),  # Thaana
    (0x900, 0x97F, "devanagari"),
    (0x980, 0x9FF, "bengali"),
    (0xA00, 0xA7F, "gurmukhi"),
    (0xA80, 0xAFF, "gujarati"),
    (0xB00, 0xB7F, "oriya"),
    (0xB80, 0xBFF, "tamil"),
    (0xC00, 0xC7F, "telugu"),
    (0xC80, 0xCFF, "kannada"),
    (0xD00, 0xD7F, "malayalam"),
    (0xD80, 0xDFF, "sinhala"),
    (0xE00, 0xE7F, "thai"),
    (0xE80, 0xEFF, "lao"),
    (0xF00, 0xFFF, "tibetan"),
    (0x1000, 0x109F, "myanmar"),
    (0x10A0, 0x10FF, "georgian"),
    (0x1100, 0x11FF, "korean"),  # Hangul Jamo
    (0x1200, 0x137F, "ethiopic"),
    (0x13A0, 0x13FF, "cherokee"),
    (0x1780, 0x17FF, "khmer"),
    (0x1800, 0x18AF, "mongolian"),
    (0x2E80, 0x2EFF, "chinese"),  # CJK Radicals Supplement
    (0x2F00, 0x2FDF, "chinese"),  # Kangxi Radicals
    (0x3000, 0x303F, "japanese"),  # CJK Symbols and Punctuation
    (0x3040, 0x309F, "japanese"),  # Hiragana
    (0x30A0, 0x30FF, "japanese"),  # Katakana
    (0x3100, 0x312F, "chinese"),  # Bopomofo
    (0x3130, 0x318F, "korean"),  # Hangul Compatibility Jamo
    (0x31C0, 0x31EF, "japanese"),  # CJK Strokes
    (0x31F0, 0x31FF, "korean"),  # Katakana Phonetic Extensions
    (0x3400, 0x4DBF, "chinese"),  # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF, "chinese"),  # CJK Unified Ideographs
    (0xA960, 0xA97F, "korean"),  # Hangul Jamo Extended-A
    (0xAC00, 0xD7A3, "korean"),  # Hangul Syllables
    (0xD7B0, 0xD7FF, "korean"),  # Hangul Jamo Extended-B
    (0xF900, 0xFAFF, "chinese"),  # CJK Compatibility Ideographs
    (0xFB50, 0xFDFF, "arabic"),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF, "arabic"),  # Arabic Presentation Forms-B
    (0xFE30, 0xFE4F, "chinese"),  # CJK Compatibility Forms
    (0x20000, 0x2A6DF, "chinese"),  # CJK Unified Ideographs Extension B
]


def _get_char_type(ch: str) -> str:
    """Gets the type of character.

    Args:
        ch (str): Input character.

    Returns:
        int: Character type.
    """
    if ch in SPACE_CHARACTERS:
        return "space"
    if ch in SEP_CHARACTERS:
        return "sep"

    try:
        cp = ord(ch)
    except:
        return "english"
    for start, end, char_type in CHAR_TYPE_RANGE_MAP:
        if start <= cp <= end:
            return char_type
    return "ignore"


def word_to_phoneme(words, pinyin_phoneme_dict):
    ph_seqs = ""
    sentence = ""
    pinyins = ""
    for word in words:
        word_type = _get_char_type(word)
        if word_type == "chinese":
            sentence += word

    if len(sentence) > 0:
        pinyins = lazy_pinyin(sentence)
        # print(f'{words} Pinyin: {pinyins}')
    else:
        return ph_seqs

    for word in words:
        if word == "SP" or word == "AP":
            ph_seqs += word + " "
        elif word in sentence:
            # print(f'Word {word} found in {sentence}')
            pinyin = pinyins.pop(0) if len(pinyins) > 0 else pinyins[-1]
            if pinyin in pinyin_phoneme_dict:
                ph_seqs += pinyin_phoneme_dict[pinyin] + " "
            else:
                ph_seqs += "SP" + " "
                print(f"Pinyin {pinyin} not found in dictionary")
        else:
            ph_seqs += "SP" + " "
    if len(pinyins) > 0:
        print(f"Pinyin {pinyins} not found in sentence")
    return ph_seqs
