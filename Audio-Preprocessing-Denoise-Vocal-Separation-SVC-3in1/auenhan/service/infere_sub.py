# Copyright (c) 2024 MusicBeing Project. All Rights Reserved.
#
# Author: Feee <cgoxopx@outlook.com>
import multiprocessing
import queue
import os
import time
import json

import logging

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

import pika
from auenhan.service.mqc_helper import SUBQ_NAME, get_mq_cnx
from auenhan.service.result_pub import send_push

from auenhan.service.remote_file import download_file_from_url, upload_file
from auenhan.utils import generate_random_string
import auenhan.service.audio_processor
import auenhan.config_loader

config = auenhan.config_loader.config

current_dir = os.path.dirname(os.path.abspath(__file__))

task_queue = multiprocessing.Queue()  
upload_queue = multiprocessing.Queue()  
error_queue = multiprocessing.Queue()  
processor = auenhan.service.audio_processor.TaskManager(task_queue, upload_queue, error_queue, config.processor.device_list)  
processor.start()  

def read_all_from_queue(q):  
    results = []  
    while True:  
        try:  
            # 尝试从队列中获取一个元素  
            item = q.get_nowait()  # 在没有元素时抛出queue.Empty异常  
            results.append(item)  
        except queue.Empty:  
            # 如果队列为空，则退出循环  
            break  
    return results

def upload_tasks(queue:multiprocessing.Queue):
    msg = read_all_from_queue(queue)

    if len(msg) == 0:
        return

    logger.info(" Done! Sending succ type msg...")
    data = {"type":"succ","msg":msg}
    send_push(data)
    logger.info("upload:%s", data)

def upload_error(queue:multiprocessing.Queue):
    msg = read_all_from_queue(queue)

    if len(msg) == 0:
        return

    logger.info(" Done! Sending error type msg...")
    data = {"type":"succ","error":msg}
    send_push(data)
    logger.info("error:%s", data)

def upload_process_func(upload_queue, error_queue):
    logger.info("upload_process_func:start")
    while True:
        # logger.info("loop")
        try:
            upload_tasks(upload_queue)
        except Exception as e:
            logger.error("upload_process_func error: %s",e)
        try:
            upload_error(error_queue)
        except Exception as e:
            logger.error("upload_process_func error: %s",e)
        time.sleep(4)

def process_audio(tasks:list[dict]):
    for task in tasks:
        logger.info("task:%s", task)
        processor.add_task(task)  

upload_process = multiprocessing.Process(target=upload_process_func, args=(upload_queue,error_queue))
upload_process.start()

def callback(
    channel: pika.channel.Channel,
    method: pika.spec.Basic.Deliver,
    properties: pika.spec.BasicProperties,
    body: bytes
):
    pid = os.getpid()
    data = json.loads(body)
    logger.info(" [x]-%d ATTN: Sending the ok msg immediately!",pid)

    result = []
    for it in data:
        result.append({"mid":it["mid"],"oss":"","server":"end"})
    send_push({"type":"ok","msg":result})
    
    logger.info(" [x]-%d Start merge data...", pid)
    # This is just a simulation of time-consuming work
    
    process_audio(data)

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
                queue=SUBQ_NAME,
                auto_ack=True,
                on_message_callback=callback)

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
    # print("start")
    logger.info("start")
    workers_num = 8
    mpp = multiprocessing.Pool(processes=workers_num)
    for i in range(workers_num):
        mpp.apply_async(subscribe)
    mpp.close()
    mpp.join()
