from pypinyin import lazy_pinyin

#######################################################################################################
# Constants
#######################################################################################################
PINYIN_TO_PHONEME_DICT_FILE = "pinyin2phoneme.txt"
SPECIAL_TOKEN_PATTERN = r"<\|.*?\|>"
SEP_CHARACTERS = ["，", "。", "？", "：", "！", "；", ",", ".", ":", ";", "?", "!", "、", "/", "\\",
                  "\r", "\n"]
SPACE_CHARACTERS = [" ", "　", "\t"]

#######################################################################################################
# Character type ranges (based on Unicode)
# arranged in (start, end, type)
#######################################################################################################
CHAR_TYPE_RANGE_MAP = [
    (0x27, 0x27, "en_alphabet"),        # '
    (0x30, 0x39, "en_digit"),           # 0-9
    (0x41, 0x5a, "en_alphabet"),        # A-Z
    (0x61, 0x7a, "en_alphabet"),        # a-z
    (0x100, 0x17f, "latin_alphabet"),   # Latin Extended-A
    (0x180, 0x24f, "latin_alphabet"),   # Latin Extended-B
    (0x370, 0x3ff, "greek"),            # Greek and Coptic
    (0x400, 0x4ff, "cyrillic"),         # Cyrillic
    (0x530, 0x58f, "armenian"),         # Armenian
    (0x590, 0x5ff, "hebrew"),           # Hebrew
    (0x600, 0x6ff, "arabic"),           # Arabic
    (0x700, 0x74f, "syriac"),           # Syriac
    (0x750, 0x77f, "arabic"),           # Arabic Supplement
    (0x780, 0x7bf, "thaana"),           # Thaana
    (0x900, 0x97f, "devanagari"),
    (0x980, 0x9ff, "bengali"),
    (0xa00, 0xa7f, "gurmukhi"),
    (0xa80, 0xaff, "gujarati"),
    (0xb00, 0xb7f, "oriya"),
    (0xb80, 0xbff, "tamil"),
    (0xc00, 0xc7f, "telugu"),
    (0xc80, 0xcff, "kannada"),
    (0xd00, 0xd7f, "malayalam"),
    (0xd80, 0xdff, "sinhala"),
    (0xe00, 0xe7f, "thai"),
    (0xe80, 0xeff, "lao"),
    (0xf00, 0xfff, "tibetan"),
    (0x1000, 0x109f, "myanmar"),
    (0x10a0, 0x10ff, "georgian"),
    (0x1100, 0x11ff, "korean"),         # Hangul Jamo
    (0x1200, 0x137f, "ethiopic"),
    (0x13a0, 0x13ff, "cherokee"),
    (0x1780, 0x17ff, "khmer"),
    (0x1800, 0x18af, "mongolian"),
    (0x2e80, 0x2eff, "chinese"),        # CJK Radicals Supplement
    (0x2f00, 0x2fdf, "chinese"),        # Kangxi Radicals
    (0x3000, 0x303f, "japanese"),       # CJK Symbols and Punctuation
    (0x3040, 0x309f, "japanese"),       # Hiragana
    (0x30a0, 0x30ff, "japanese"),       # Katakana
    (0x3100, 0x312f, "chinese"),        # Bopomofo
    (0x3130, 0x318f, "korean"),         # Hangul Compatibility Jamo
    (0x31c0, 0x31ef, "japanese"),       # CJK Strokes
    (0x31f0, 0x31ff, "korean"),         # Katakana Phonetic Extensions
    (0x3400, 0x4dbf, "chinese"),        # CJK Unified Ideographs Extension A
    (0x4e00, 0x9fff, "chinese"),        # CJK Unified Ideographs
    (0xa960, 0xa97f, "korean"),         # Hangul Jamo Extended-A
    (0xac00, 0xd7a3, "korean"),         # Hangul Syllables
    (0xd7b0, 0xd7ff, "korean"),         # Hangul Jamo Extended-B
    (0xf900, 0xfaff, "chinese"),        # CJK Compatibility Ideographs
    (0xfb50, 0xfdff, "arabic"),         # Arabic Presentation Forms-A
    (0xfe70, 0xfeff, "arabic"),         # Arabic Presentation Forms-B
    (0xfe30, 0xfe4f, "chinese"),        # CJK Compatibility Forms
    (0x20000, 0x2a6df, "chinese"),      # CJK Unified Ideographs Extension B
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
    pinyins=''
    for word in words:
        word_type = _get_char_type(word)
        if word_type == 'chinese':
            sentence += word
    
    if len(sentence) > 0:
        pinyins = lazy_pinyin(sentence)
        # print(f'{words} Pinyin: {pinyins}')
    else:
        return ph_seqs
    
    for word in words:
        if word == 'SP' or word == 'AP':
            ph_seqs += word + " "
        elif word in sentence:
            # print(f'Word {word} found in {sentence}')
            pinyin = pinyins.pop(0) if len(pinyins) > 0 else pinyins[-1]
            if pinyin in pinyin_phoneme_dict:
                ph_seqs += pinyin_phoneme_dict[pinyin] + " "
            else:
                ph_seqs += 'SP' + " "
                print(f'Pinyin {pinyin} not found in dictionary')
        else:
            ph_seqs += 'SP' + " "
    if len(pinyins) > 0:
        print(f'Pinyin {pinyins} not found in sentence')
    return ph_seqs