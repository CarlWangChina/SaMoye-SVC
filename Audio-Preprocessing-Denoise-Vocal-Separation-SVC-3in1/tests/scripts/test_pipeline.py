# Copyright (c) 2024 MusicBeing Project. All Rights Reserved.
#
# Author: Feee <cgoxopx@outlook.com>
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from auenhan.service.audio_processor import *

if __name__ == "__main__":  
    # 创建任务队列和（可选的）结果队列  
    task_queue = multiprocessing.Queue()  
    result_queue = multiprocessing.Queue()  # 如果不需要结果队列，可以注释掉这行  
    error_queue = multiprocessing.Queue()
    # result_queue = None  # 设置为None表示不返回结果  
  
    # 创建任务管理器并启动进程  
    manager = TaskManager(task_queue, result_queue, error_queue, num_processes=3)  
    manager.start()  
  
    # 添加一些任务  
    manager.add_task({})  
    
    print("push task success")
  
    while True:  
        print(result_queue.get())
        break

    print("read result success")
            
    # 等待一段时间，以便进程可以处理任务  
    time.sleep(2)  
  
    # 停止进程  
    manager.stop()  
    print("success")