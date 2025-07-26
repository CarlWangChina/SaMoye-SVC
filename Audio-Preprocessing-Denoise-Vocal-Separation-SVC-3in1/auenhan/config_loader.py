from omegaconf import OmegaConf
import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+'/..')

if os.path.exists("/configs/config.yaml"): # for docker
    config = OmegaConf.load("/configs/config.yaml")
elif os.path.exists(PROJECT_ROOT+"/configs/config.yaml"):
    config = OmegaConf.load(PROJECT_ROOT+"/configs/config.yaml")

# 配置 OSS 访问参数
END_POINT = config.mq.END_POINT
ACCESS_KEY_ID = config.mq.ACCESS_KEY_ID
ACCESS_KEY_SECRET = config.mq.ACCESS_KEY_SECRET
BUCKET_NAME = config.mq.BUCKET_NAME
