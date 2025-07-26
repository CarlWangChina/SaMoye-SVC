import pika

import auenhan.config_loader

# Message Queue Connection Parameters
MQCP_HOST = auenhan.config_loader.config.mq.MQCP_HOST
MQCP_PORT = auenhan.config_loader.config.mq.MQCP_PORT 
MQCP_VHOST = auenhan.config_loader.config.mq.MQCP_VHOST
MQCP_USER = auenhan.config_loader.config.mq.MQCP_USER 
MQCP_PASS = auenhan.config_loader.config.mq.MQCP_PASS 
SUBQ_NAME = auenhan.config_loader.config.mq.SUBQ_NAME 
PUBQ_NAME = auenhan.config_loader.config.mq.PUBQ_NAME 

def get_mq_cnx():
    credentials = pika.PlainCredentials(
        username=MQCP_USER,
        password=MQCP_PASS,
        erase_on_connect=True)
    cparameters = pika.ConnectionParameters(
        host=MQCP_HOST,
        port=MQCP_PORT,
        virtual_host=MQCP_VHOST,
        credentials=credentials)
    return pika.BlockingConnection(cparameters)
