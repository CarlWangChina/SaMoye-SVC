import multiprocessing
import os
import time
import json
import subprocess
import pika

from utils import get_logger, get_hparams, get_mq_connection, mq_send_push
from .mq_callback import svc_ack, svc_generate

os.environ["TOKENIZERS_PARALLELISM"] = "false"

mq_hparams = get_hparams()["mq"]

logger = get_logger(__name__)


def style_callback(
    channel: pika.channel.Channel,
    method: pika.spec.Basic.Deliver,
    properties: pika.spec.BasicProperties,
    body: bytes,
):
    try:
        task_info = json.loads(body.decode())
    except json.JSONDecodeError as _:
        logger.info(f"get task in wrong format: {body.decode()}")
        return

    # task_info = [
    #     {"mid":"123asdf","oss":"人声MP3","gender":"m|w"},
    #     {"mid":"123asdf","oss":"人声MP3","gender":"m|w"},
    # ]

    logger.info(f"get task info: {task_info}")
    ok_msg = {
        "type": "ok",
        "msg": list(map(svc_ack, task_info)),
    }

    res_queue_name = mq_hparams["queues"]["task_ack_queue"]
    mq_send_push(
        ok_msg,
        qname=res_queue_name,
    )
    logger.info(f"ack task msg: {ok_msg}")

    succ_msg = {
        "type": "succ",
        "msg": list(map(svc_generate, task_info)),
    }
    mq_send_push(
        succ_msg,
        qname=res_queue_name,
    )
    logger.info(f"task succ msg: {succ_msg}")


def subscribe():

    cnx_rec_max_t = 4
    cnx_rec_times = 0

    while cnx_rec_times <= cnx_rec_max_t:
        try:
            # don't forget to add param `--network="host"` when running docker container
            connection = get_mq_connection()

            # channel for input, sub for the task messages
            chan_ip = connection.channel()

            chan_ip.queue_declare(
                queue=mq_hparams["queues"]["task_get_queue"], durable=True
            )

            chan_ip.basic_qos(prefetch_count=1)
            chan_ip.basic_consume(
                queue=mq_hparams["queues"]["task_get_queue"],
                auto_ack=True,
                on_message_callback=style_callback,
            )

            chan_ip.start_consuming()
        # Don't recover if connection was closed by broker
        except pika.exceptions.ConnectionClosedByBroker as e:
            logger.error(e)
            break
        # Don't recover on channel errors
        except pika.exceptions.AMQPChannelError as e:

            cnx_rec_times += 1
            logger.error(e)
        # Recover on all other connection errors
        except pika.exceptions.AMQPConnectionError as e:
            logger.error(e)
            cnx_rec_times += 1
            # time.sleep(2)
            continue


def start():
    # # TODO read from config file
    # workers_num = 8
    # mpp = multiprocessing.Pool(processes=workers_num)
    # for i in range(workers_num):
    #     mpp.apply_async(subscribe)
    # mpp.close()
    # mpp.join()
    subscribe()
