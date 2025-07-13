import multiprocessing
import os
import random
import time

import pika
from .mqc_helper import SUBQ_NAME, get_mq_cnx
from .result_pub import send_push


def callback(
    channel: pika.channel.Channel,
    method: pika.spec.Basic.Deliver,
    properties: pika.spec.BasicProperties,
    body: bytes
):
    pid = os.getpid()
    print(f" [x]-{pid} Received body: {body.decode()}")
    print(f" [x]-{pid} ATTN: Sending the ok msg immediately!")
    # TODO modify the dict for your service
    send_push({"type":"ok","msg":[{"mid":"xxx"}]})
    # TODO repalce the your real work entrace here
    

    print(f" [x]-{pid} Start doing your work here...")
    # This is just a simulation of time-consuming work
    time.sleep(random.randint(3, 9))
    print(f" [x]-{pid} Done! Sending succ type msg...")
    # TODO again, modify the dict should be returned by your service
    send_push({"type":"succ","msg":[{"mid":"xxx", "oss":"the generated lrc..."}]})

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
    # TODO read from config file
    workers_num = 8
    mpp = multiprocessing.Pool(processes=workers_num)
    for i in range(workers_num):
        mpp.apply_async(subscribe)
    mpp.close()
    mpp.join()
