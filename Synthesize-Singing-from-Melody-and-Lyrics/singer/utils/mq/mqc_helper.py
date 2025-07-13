import pika

# Message Queue Connection Parameters
MQCP_HOST = 'rabbitmq-serverless-cn-jeo3og3a00f.cn-zhangjiakou.amqp-2.net.mq.amqp.aliyuncs.com'
MQCP_PORT = 5672
MQCP_VHOST = 'caichong'
MQCP_USER = 'MjpyYWJiaXRtcS1zZXJ2ZXJsZXNzLWNuLWplbzNvZzNhMDBmOkxUQUk1dFFtZnVnOFJKUlV2aTdmUE5XZw=='
MQCP_PASS = 'RUY1NTQ2NUM3MDc0RUVBRERGOTEzQkI3RUNCNjZDOTYyODlGQkQ4NDoxNzExOTg1ODA3Mjk5'

# Queue Names to your service
SUBQ_NAME = 'create_music_taskend'
PUBQ_NAME = 'create_music_taskend_resp'

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