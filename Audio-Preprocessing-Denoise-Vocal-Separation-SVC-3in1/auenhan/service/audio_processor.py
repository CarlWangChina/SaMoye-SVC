# Copyright (c) 2024 MusicBeing Project. All Rights Reserved.
#
# Author: Feee <cgoxopx@outlook.com>
import multiprocessing  
import queue  
import time  
import os  
from auenhan.separation import DemuxExtractor, UVRExtractor
import auenhan.config_loader as config
import logging
from auenhan.service.remote_file import download_file_from_url, download_file, upload_file
from auenhan.utils import generate_random_string

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

class TaskProcessor(multiprocessing.Process):  
    def __init__(self, task_queue: multiprocessing.Queue, result_queue: multiprocessing.Queue = None, error_queue: multiprocessing.Queue = None, device:int=0):  
        super().__init__()  
        self.task_queue = task_queue  
        self.result_queue = result_queue  
        self.error_queue = error_queue  
        self.stop_event = multiprocessing.Event()  
        self.device = f"cuda:{device}"
        if config.config.processor.model=="demucs":
            self.extractor = DemuxExtractor(device=self.device, use_denoise=config.config.processor.use_denoise)
        elif config.config.processor.model=="uvr":
            self.extractor = UVRExtractor(model=config.processor.uvr_path, device=self.device)
  
    def run(self):  
        while not self.stop_event.is_set():  
            task = ""
            try:  
                # 阻塞等待获取任务  
                task = self.task_queue.get(block=True, timeout=1)  
                if task is None:  # 收到停止信号  
                    break  
                # 调用函数处理任务  
                self.process_task(task)  
            except queue.Empty:  
                # 队列为空，继续等待  
                pass  
            except Exception as e:
                logger.error("process task error:%s", e)
  
    def process_task(self, task): 
        # tasks = [
        #     {
        #         "mid":"", //music ID
        #         "spk":",//音色值（liudehua）
        #         "org_music_oss":""
        #     }
        # ]
        mid = task["mid"]
        spk = task["spk"]
        org_music_oss = task["org_music_oss"]
        local_path = f"/tmp/{mid}_ori.mp3"
        acc_path = f"/tmp/{mid}_acc.mp3"
        vocal_path = f"/tmp/{mid}_vocal.mp3"
        vocal_svc_path = f"/tmp/{mid}_vocal_svc.mp3"
        rand_id = generate_random_string(16)
        vocal_svc_oss = f"auenhan/{rand_id}/{mid}_vocal.mp3"
        acc_oss = f"auenhan/{rand_id}/{mid}_acc.mp3"
        try:
            if org_music_oss.startswith("http"):
                download_file_from_url(org_music_oss, local_path)
            else:
                download_file(org_music_oss, local_path)

            self.extractor.process_file(infile=local_path, outfile_vocal=vocal_path, outfile_acc=acc_path)

            # 如果有结果队列，则将结果放入  
            if self.result_queue:  
                upload_file(vocal_svc_path, vocal_svc_oss)
                upload_file(acc_path, acc_oss)
                self.result_queue.put({
                    "mid":mid,
                    "svc_oss":f"https://{config.BUCKET_NAME}.oss-cn-zhangjiakou.aliyuncs.com/{vocal_svc_oss}",
                    "acc_oss":f"https://{config.BUCKET_NAME}.oss-cn-zhangjiakou.aliyuncs.com/{acc_oss}"
                })  
        except Exception as e:
            logger.error("process file error:%s", e)
            self.error_queue.put(f"api_call:{e},{task}")
        finally:
            os.remove(local_path)
            os.remove(acc_path)
            os.remove(vocal_path)
            os.remove(vocal_svc_path)
            

    def stop(self):  
        """发送停止信号"""  
        self.stop_event.set()  
        self.task_queue.put(None)  # 放入一个None作为停止信号  
  
  
class TaskManager:  
    def __init__(self, task_queue: multiprocessing.Queue, result_queue: multiprocessing.Queue = None, error_queue: multiprocessing.Queue = None, devices: list[int] = [0,1,2,3]):  
        self.task_queue = task_queue  
        self.result_queue = result_queue  
        self.error_queue = error_queue  
        self.devices = devices
        self.processors = [TaskProcessor(self.task_queue, self.result_queue, self.error_queue, device) for device in self.devices]  
  
    def start(self):  
        """启动多个进程"""  
        for p in self.processors:  
            p.start()  
  
    def stop(self):  
        """发送停止信号并等待所有进程结束"""  
        for p in self.processors:  
            p.stop()  
        for p in self.processors:  
            p.join()  
  
    def add_task(self, task):  
        """添加任务到任务队列"""  
        self.task_queue.put(task)  
