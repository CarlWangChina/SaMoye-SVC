from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

from ds_to_midi import ds_to_midi
from download_from_s3 import download_from_index_music
from midi_lyric_align import midi_lyric_align_main
from generate_vocal import generate_vocal
from remix import remix_main
from change_ds_midi import change_ds_midi
from concurrent.futures import ThreadPoolExecutor
import asyncio
import logging
from logging.handlers import RotatingFileHandler

# 创建一个logger  
logger = logging.getLogger(__name__)  
  
logging.basicConfig(level=logging.INFO,  
                    format='%(asctime)s - %(levelname)s - %(message)s',  
                    handlers=[RotatingFileHandler('my_app.log', maxBytes=1024*1024, backupCount=5),  
                              logging.StreamHandler()])   

app = FastAPI()
executor = ThreadPoolExecutor()
class TestMessage(BaseModel):
    test_name: str
    test_midi_file: str
    spk: list
    batch_size: int
    CUDA: int
    modified: int
    overwrite: int


@app.post("/download_from_s3/")
async def fa_download_from_s3(test_inputs: TestMessage):
    # 设置根目录
    root_dir = "/export/data/home/john/MuerSinger2"
    # 更改工作目录到根目录
    os.chdir(root_dir)
    try:
        download_from_index_music(test_inputs.test_name,rewrite=False)
        # 记录处理成功的日志
        logging.info(f"Download from s3 for test {test_inputs.test_name} successful")
        return {"message": "Download successful"}
    except Exception as e:
        # 记录处理失败的日志
        logging.error(f"Download from s3 for test {test_inputs.test_name} failed, error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")

@app.post("/get_ds/")
async def fa_midi_lyric_align(test_inputs: TestMessage):
    # 设置根目录
    root_dir = "/export/data/home/john/MuerSinger2"
    # 更改工作目录到根目录
    os.chdir(root_dir)
    try:
        midi_lyric_align_main(test_inputs.test_name, test_inputs.test_midi_file, test_inputs.modified)
        # 记录处理成功的日志  
        logging.info(f"ds generation for test {test_inputs.test_midi_file} successful")  
        return {"message": "ds generation successful"}
    except Exception as e:
        # 记录处理失败的日志
        logging.error(f"ds generation for test {test_inputs.test_midi_file} failed, error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ds generation Error occurred: {str(e)}")

@app.post("/convert_ds_to_midi/")
async def fa_convert_ds_to_midi(test_inputs: TestMessage):
    # 设置根目录
    root_dir = "/export/data/home/john/MuerSinger2"
    # 更改工作目录到根目录
    os.chdir(root_dir)
    try:
        ds_to_midi(test_inputs.test_name, test_inputs.test_midi_file)
        # 记录处理成功的日志
        logging.info(f"ds_to_midi for test {test_inputs.test_midi_file} successful")
        return {"message": "Conversion successful"}
    except Exception as e:
        # 记录处理失败的日志
        logging.error(f"ds_to_midi for test {test_inputs.test_midi_file} failed, error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ds_to_midi Error occurred: {str(e)}")

# @app.post("/change_ds_midi/")
# async def fa_change_ds_midi(test_inputs: TestMessage):
#     # 设置根目录
#     root_dir = "/export/data/home/john/MuerSinger2"
#     # 更改工作目录到根目录
#     os.chdir(root_dir)
#     try:
#         await change_ds_midi(test_inputs)
#         # 记录处理成功的日志
#         logging.info(f"change_ds_midi for test {test_inputs.test_midi_file} successful")
#         return {"message": "Change successful"}
#     except Exception as e:
#         # 记录处理失败的日志
#         logging.error(f"change_ds_midi for test {test_inputs.test_midi_file} failed, error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"change_ds_midi Error occurred: {str(e)}")
# 异步函数
async def fa_change_ds_midi(test_inputs: TestMessage):
    # 设置根目录
    root_dir = "/export/data/home/john/MuerSinger2"
    # 更改工作目录到根目录
    os.chdir(root_dir)
    try:
        await change_ds_midi(test_inputs)
        # 记录处理成功的日志
        logging.info(f"change_ds_midi for test {test_inputs.test_midi_file} successful")
        return {"message": "Change successful"}
    except Exception as e:
        # 记录处理失败的日志
        logging.error(f"change_ds_midi for test {test_inputs.test_midi_file} failed, error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"change_ds_midi Error occurred: {str(e)}")
    
# 定义路由
@app.post("/change_ds_midi/")
async def change_ds_midi_handler(test_inputs: TestMessage):
    # 异步运行处理函数
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(executor, fa_change_ds_midi, test_inputs)
    return result

# 异步函数
async def fa_generate_vocal(test_inputs: TestMessage):
    # 设置根目录
    root_dir = "/export/data/home/john/MuerSinger2/DiffSinger"
    # 更改工作目录到根目录
    os.chdir(root_dir)
    try:
        await generate_vocal(test_inputs)
        # 记录处理成功的日志
        logging.info(f"generate_vocal for test {test_inputs.test_midi_file} successful")
        return {"message": "Generation successful"}
    except Exception as e:
        # 记录处理失败的日志
        logging.error(f"generate_vocal for test {test_inputs.test_midi_file} failed, error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"generate_vocal Error occurred: {str(e)}")

# 定义路由
@app.post("/generate_vocal/")
async def generate_vocal_handler(test_inputs: TestMessage):
    # 异步运行处理函数
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(executor, fa_generate_vocal, test_inputs)
    return result

# @app.post("/generate_vocal/")
# async def fa_generate_vocal(test_inputs: TestMessage):
#     # 设置根目录
#     root_dir = "/export/data/home/john/MuerSinger2/DiffSinger"
#     # 更改工作目录到根目录
#     os.chdir(root_dir)
#     try:
#         await generate_vocal(test_inputs)
#         # 记录处理成功的日志
#         logging.info(f"generate_vocal for test {test_inputs.test_midi_file} successful")
#         return {"message": "Generation successful"}
#     except Exception as e:
#         # 记录处理失败的日志
#         logging.error(f"generate_vocal for test {test_inputs.test_midi_file} failed, error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"generate_vocal Error occurred: {str(e)}")
    
@app.post("/mix_song/")
async def fa_mix_song(test_inputs: TestMessage):
    # 设置根目录
    root_dir = "/export/data/home/john/MuerSinger2"
    # 更改工作目录到根目录
    os.chdir(root_dir)
    try:
        remix_main(test_inputs)
        # 记录处理成功的日志
        logging.info(f"mix_song for test {test_inputs.test_midi_file} successful")
        return {"message": "Mixing successful"}
    except Exception as e:
        # 记录处理失败的日志
        logging.error(f"mix_song for test {test_inputs.test_midi_file} failed, error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"mix_song Error occurred: {str(e)}")
    
# @app.post("/midi_to_vocal/")
# async def fa_midi_to_vocal(test_inputs: TestMessage):
#     try:
#         midi_lyric_align(test_inputs.test_name, test_inputs.test_midi_file)
#         generate_vocal(test_inputs)
#         remix_main(test_inputs)
#         return {"message": "All successful"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")

