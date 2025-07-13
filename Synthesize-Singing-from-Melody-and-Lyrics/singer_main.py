import multiprocessing
import os
import json
import time

from pathlib import Path

import pretty_midi
import pika

from singer.utils.logging import get_logger
from singer.utils.mq.mqc_helper import SUBQ_NAME, get_mq_cnx
from singer.utils.mq.result_pub import send_push
from singer.preprocess.preprocess_pipeline import preprocess_main
from singer.test_pipeline_aces2ds import pipline_aces2ds_main
from singer.utils.mq.mq_transfer_data import (
    download_file_from_url,
    upload_file,
    merge_oss_address,
)

logger = get_logger(__name__)
root_path = Path(__file__).parent.resolve()

def test_main(test_id="9768ade5ad75ad9777c9bfc6d2505130"):
    pid = os.getpid()
    print(f" [x]-{pid} Received body: ")
    print(f" [x]-{pid} ATTN: Sending the ok msg immediately!")
    # 接收内容body：[{mid:"123asdf","midioss":"oss地址","new_midioss":"oss地址","lyricoss":"oss地址"}]
    # 1. song_id: 歌曲id对应的内容是mid
    # 2. json ：时间戳、cccr，对应的内容是lyricoss,timeoss
    # 3. mid：歌曲新的midi文件，对应的内容是midioss

    # Load JSON data
    # 自动获取根目录
    
    try:
        test_txt = Path(f"/home/john/DuiniutanqinSinger/data/test_data/{test_id}.txt")
        test_dir = Path(f"/home/john/DuiniutanqinSinger/data/test_data/{test_id}")
        if test_txt.exists():
            with open(
                test_txt,
                "r",
                encoding="utf-8",
            ) as f:
                body = f.read()
            message_body = json.loads(body)
            # print(f" [x]-{pid} Received message_body: {message_body}")

            song_id = message_body[0]["mid"]
            # print(f" [x]-{pid} Received song_id: {song_id}")
            timeoss = message_body[0]["oss"]
            # print(f" [x]-{pid} Received timeoss: {timeoss}")
            lyricoss = message_body[0]["lyricoss"]
            # print(f" [x]-{pid} Received lyricoss: {lyricoss}")
            midoss = message_body[0]["oldMidiOss"]
            # print(f" [x]-{pid} Received midoss: {midoss}")
            new_midoss = message_body[0]["midioss"]
            # print(f" [x]-{pid} Received song_id: {song_id}")
            time_json = json.loads(timeoss)
            midi_temp_path = root_path / Path(f"data/midi/temp/{song_id}_origin.mid")
            new_midi_temp_path = root_path / Path(f"data/midi/temp/{song_id}_new.mid")
            download_file_from_url(midoss, midi_temp_path)
            download_file_from_url(new_midoss, new_midi_temp_path)

        elif test_dir.exists():
            song_id = test_id
            with open(test_dir / f"{test_id}_fixed.json", "r", encoding="utf-8") as f:
                time_json = json.load(f)
            lyricoss = None
            midi_temp_path = test_dir / f"{test_id}_origin.mid"
            new_midi_temp_path = test_dir / "new_1" / f"{test_id}_new_1.mid"
        else:
            print(f" [x]-{pid} No such file or directory: {test_id}")
            return 0

    except json.JSONDecodeError as e:
        print(f" [x]-{pid} Error decoding JSON: {e}")
        return 0

    midi = pretty_midi.PrettyMIDI(str(midi_temp_path))
    new_midi = pretty_midi.PrettyMIDI(str(new_midi_temp_path))
    # send_push(
    #     {
    #         "type": "ok",
    #         "msg": [
    #             {
    #                 "mid": str(song_id),
    #                 "oss": "",
    #                 "keyword": "",
    #                 "title": "",
    #                 "server": "end",
    #             }
    #         ],
    #     }
    # )

    print(f" [x]-{song_id} Start doing your work here...")
    # 任务流程：
    # 1.1chang_lyric: 将json中的歌词换成新的歌词
    # 1.2midi_lyric_align: 将 json 中的时间戳、cccr转换为 ds 文件
    # 2. 根据这个 midi chang_ds_midi 为相应midi的 ds
    # 3. 根据这个 ds 调用 variance 模型进行推理
    # 4. 调用acoustic模型进行推理
    # try:
    # 设置根目录,自动获取当前 parent 目录
    vocal_temp_path = preprocess_main(
        song_id, time_json, lyricoss, midi, new_midi, root_path
    )
    print(f" [x]-{pid} Done! vocal_temp_path: {vocal_temp_path}")
    # pipline_aces2ds_main(song_id, time_json, lyricoss, midi, new_midi, root_path)
    # except Exception as e:
    #     print(f" [x]-{pid} Error processing file: {e}")
    #     return 0

    print(f" [x]-{pid} Done! Sending succ type msg...")
    # 返回内容：
    # 1. vocal.wav：新的vocal文件
    # 2. mix.wav：（可选）混合伴奏后的mix文件
    mp3_oss_address = f"test/{song_id}.mp3"  # 上传到 OSS 的地址
    # upload_file(vocal_temp_path / Path(f"{song_id}.mp3"), mp3_oss_address)

    total_mp3_oss_address = merge_oss_address(mp3_oss_address)
    print(f"Total mp3 oss address: {total_mp3_oss_address}")


if __name__ == "__main__":
    while True:
        # 键盘输入songid
        songid = input("Please input songid: ")
        test_main(songid)

    # message_body = [
    #     {
    #         "mid": "100939",
    #         "new_midioss": "/home/john/DuiniutanqinSinger/data/test_data/100939/new_1/100939_new_1.mid",
    #         "midioss": "/home/john/DuiniutanqinSinger/data/test_data/100939/100939_origin.mid",
    #         "lyricoss": "/home/john/DuiniutanqinSinger/data/test_data/100939/100939_fixed.json",
    #     }
    # ]
