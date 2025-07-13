import re

from .utils.lyric_to_phoneme import _get_char_type

lyrics = {"一二三四五六七八九十"}
lyrics = list(lyrics.pop())


def mq_clean_lyric(json_data):
    """
    清洗数据，只保留C段

    Returns:
        list: json数据

    """
    new_json_data = []
    for record in json_data:
        if "C" in record["cccr"]:
            # TODO: 明天跟一库对完再改
            # new_record = {}
            # new_record["words"] = []
            # for i, flag in enumerate(record["cccr"]):
            #     if flag == "C":
            #         new_record["words"].append(record["words"][i])
            # new_record["text"] = "".join([word[2] for word in new_record["words"]])
            # new_record["start"] =
            new_json_data.append(record)
    return new_json_data


def mq_change_lyric(json_data, new_lyric):
    """
    将json中的歌词换成新的歌词

    Args:
        json_data (list): json数据
        new_lyric (str): 新歌词

    Returns:
        list: json数据

    """
    # 去除特殊符号后，按字分割成列表
    new_lyric_lists = [
        re.findall(r"[\u4e00-\u9fff]|[a-zA-Z]+", line) for line in new_lyric.split("\n")
    ]
    # print(new_lyric_lists)

    new_lyric_lists_len = len(new_lyric_lists)

    for i, record in enumerate(json_data):
        words = record["words"]
        new_lyric_list = new_lyric_lists[i % new_lyric_lists_len]
        new_lyric_list_len = len(new_lyric_list)
        word_count = 0
        for j, word in enumerate(words):
            word_type = _get_char_type(word[2])
            if (
                word_type == "chinese"
                or word_type == "english"
                or word_type == "korean"
                or word_type == "japanese"
            ):
                words[j][2] = new_lyric_list[word_count % new_lyric_list_len]
                word_count += 1
        record["text"] = "".join([word[2] for word in words])
    # print(json_data)
    return json_data
