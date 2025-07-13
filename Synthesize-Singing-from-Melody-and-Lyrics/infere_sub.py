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
# from singer.preprocess.change_lyric import mq_change_lyric
# from singer.preprocess.midi_lyric_align import mq_midi_lyric_align
# from singer.preprocess.change_ds_midi import mq_change_note
# from singer.preprocess.generate_vocal import mq_run_diffsinger_command
# from singer.preprocess.utils.wav_2_mp3 import convert_wav_to_mp3
from singer.utils.mq.mq_transfer_data import (
    download_file_from_url,
    upload_file,
    merge_oss_address,
)
from singer.preprocess.preprocess_pipeline import preprocess_main

logger = get_logger(__name__)


def callback(
    channel: pika.channel.Channel,
    method: pika.spec.Basic.Deliver,
    properties: pika.spec.BasicProperties,
    body: bytes,
):
    """
    The callback function for the message queue, revised by Yongsheng Feng

    Args:
        channel (pika.channel.Channel): The channel object
        method (pika.spec.Basic.Deliver): The method object
        properties (pika.spec.BasicProperties): The properties object
        body (bytes): The body of the message

    Returns:
        None
    """
    pid = os.getpid()
    print(f" [x]-{pid} Received body: {body.decode()}")
    print(f" [x]-{pid} ATTN: Sending the ok msg immediately!")
    # 接收内容body：[{mid:"123asdf","midioss":"oss地址","new_midioss":"oss地址","lyricoss":"oss地址"}]
    # 1. song_id: 歌曲id对应的内容是mid
    # 2. json ：时间戳、cccr，对应的内容是lyricoss,timeoss
    # 3. mid：歌曲新的midi文件，对应的内容是midioss
    # Decode body and replace newline characters
    # Load JSON data
    try:
        message_body = json.loads(body)
        print(f" [x]-{pid} Received message_body: {message_body}")

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

    except json.JSONDecodeError as e:
        print(f" [x]-{pid} Error decoding JSON: {e}")
        return 0

    root_path = Path("/home/john/DuiniutanqinSinger/")

    midi_temp_path = root_path / Path(f"data/midi/temp/{song_id}_origin.mid")
    new_midi_temp_path = root_path / Path(f"data/midi/temp/{song_id}_new.mid")

    # 下载文件
    try:
        download_file_from_url(midoss, midi_temp_path)
        download_file_from_url(new_midoss, new_midi_temp_path)

        time_json = json.loads(timeoss)
        # print(f" [x]-{pid} Received time_json: {time_json}")
        midi = pretty_midi.PrettyMIDI(str(midi_temp_path))
        new_midi = pretty_midi.PrettyMIDI(str(new_midi_temp_path))
    except Exception as e:
        logger.error(" [x]-%s Error downloading file: %s", pid, e)
        print(f" [x]-{pid} Error downloading file: {e}")
        return 0

    send_push(
        {
            "type": "ok",
            "msg": [
                {
                    "mid": str(song_id),
                    "oss": "",
                    "keyword": "",
                    "title": "",
                    "server": "end",
                }
            ],
        }
    )

    print(f" [x]-{song_id} Start doing your work here...")
    # 任务流程：
    # 1.1chang_lyric: 将json中的歌词换成新的歌词
    # 1.2midi_lyric_align: 将 json 中的时间戳、cccr转换为 ds 文件
    # 2. 根据这个 midi chang_ds_midi 为相应midi的 ds
    # 3. 根据这个 ds 调用 variance 模型进行推理
    # 4. 调用acoustic模型进行推理
    try:
        vocal_temp_path = preprocess_main(
            song_id, time_json, lyricoss, midi, new_midi, root_path
        )
        print(f" [x]-{pid} Done! vocal_temp_path: {vocal_temp_path}")
        # # 设置根目录
        # root_dir = "/export/data/home/john/DuiniutanqinSinger/"
        # # 更改工作目录到根目录
        # os.chdir(root_dir)
        # new_time_json = mq_change_lyric(time_json, lyricoss)
        # ds_data = mq_midi_lyric_align(str(song_id), midi, new_midi, new_time_json)
        # # print(f"ds_data: {ds_data}")
        # ds_data_new = mq_change_note(str(song_id), ds_data, midi)

        # ds_temp_path = root_path / Path(f"data/ds/temp/input/{song_id}.ds")
        # ds_temp_output_path = root_path / Path(f"data/ds/temp/output/{song_id}.ds")
        # vocal_temp_path = root_path / Path("data/vocal/temp/")

        # # 创建文件夹
        # ds_temp_path.parent.mkdir(parents=True, exist_ok=True)
        # with open(ds_temp_path, "w", encoding="utf-8") as f:
        #     json.dump(ds_data_new, f, ensure_ascii=False, indent=4)
        # mq_run_diffsinger_command(
        #     ds_input_path=ds_temp_path,
        #     ds_output_path=ds_temp_output_path,
        #     vocal_path=vocal_temp_path,
        # )
        # convert_wav_to_mp3(
        #     vocal_temp_path / Path(f"{song_id}.wav"),
        #     vocal_temp_path / Path(f"{song_id}.mp3"),
        # )
    except Exception as e:
        logger.error(" [x]-%s Error processing file: %s", pid, e)
        print(f" [x]-{pid} Error processing file: {e}")
        return 0

    print(f" [x]-{pid} Done! Sending succ type msg...")
    # 返回内容：
    # 1. vocal.wav：新的vocal文件
    # 2. mix.wav：（可选）混合伴奏后的mix文件
    mp3_oss_address = f"test/{song_id}.mp3"  # 上传到 OSS 的地址
    upload_file(vocal_temp_path / Path(f"{song_id}.mp3"), mp3_oss_address)

    total_mp3_oss_address = merge_oss_address(mp3_oss_address)
    print(f"Total mp3 oss address: {total_mp3_oss_address}")
    logger.info(" [x]-%s Sending succ type msg to %s", pid, total_mp3_oss_address)
    send_push(
        {
            "type": "succ",
            "msg": [
                {
                    "mid": str(song_id),
                    "oss": total_mp3_oss_address,
                    "keyword": "",
                    "title": "???",
                    "server": "end",
                }
            ],
        }
    )


def subscribe():
    # TODO read from config file
    cnx_rec_max_t = 4
    cnx_rec_times = 0

    while cnx_rec_times <= cnx_rec_max_t:
        try:
            # don't forget to add param `--network="host"` when running docker container
            connection = get_mq_cnx()

            # channel for input, sub for the task messages
            chan_ip = connection.channel()

            chan_ip.queue_declare(queue=SUBQ_NAME, durable=True)

            chan_ip.basic_qos(prefetch_count=1)
            chan_ip.basic_consume(
                queue=SUBQ_NAME, auto_ack=True, on_message_callback=callback
            )

            chan_ip.start_consuming()
        # Don't recover if connection was closed by broker
        except pika.exceptions.ConnectionClosedByBroker:
            break
        # Don't recover on channel errors
        except pika.exceptions.AMQPChannelError:
            break
        # Recover on all other connection errors
        except pika.exceptions.AMQPConnectionError:
            cnx_rec_times += 1
            time.sleep(2)
            continue


def start():
    # TODO read from config file
    workers_num = 8
    mpp = multiprocessing.Pool(processes=workers_num)
    for i in range(workers_num):
        mpp.apply_async(subscribe)
    mpp.close()
    mpp.join()
