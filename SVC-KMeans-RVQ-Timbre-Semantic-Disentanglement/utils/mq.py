import pika
import json
from .hparams import get_hparams


# mq_hparams = get_hparams()["mq"]


def get_mq_connection():
    credentials = pika.PlainCredentials(
        username=mq_hparams["username"],
        password=mq_hparams["password"],
        erase_on_connect=True,
    )
    return pika.BlockingConnection(
        pika.ConnectionParameters(
            host=mq_hparams["host"],
            port=mq_hparams["port"],
            virtual_host=mq_hparams["virtual_host"],
            credentials=credentials,
        )
    )


def mq_send_push(msg: dict, qname: str):
    connection = get_mq_connection()
    chan_op = connection.channel()
    chan_op.queue_declare(queue=qname, durable=True)
    chan_op.basic_publish(
        exchange="",
        routing_key=qname,
        body=json.dumps(msg, ensure_ascii=False, separators=(",", ":")),
        properties=pika.BasicProperties(
            delivery_mode=pika.DeliveryMode.Persistent),
    )
    chan_op.close()
    connection.close()
