import os
import subprocess
import argparse
from src.params import song_ids
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import logging
from logging.handlers import RotatingFileHandler

# 创建一个logger  
logger = logging.getLogger(__name__)  
  
logging.basicConfig(level=logging.INFO,  
                    format='%(asctime)s - %(levelname)s - %(message)s',  
                    handlers=[RotatingFileHandler('my_app.log', maxBytes=1024*1024, backupCount=5),  
                              logging.StreamHandler()])

root_path = '/export/data/home/john/MuerSinger2'

async def run_diffsinger(file_name, test_name, spk, batch_size=1, CUDA=7, overwrite=0):
    # 设置环境变量  
    env = os.environ.copy()  
    env['CUDA_VISIBLE_DEVICES'] = str(CUDA)

    async def run_subprocess(loc):
        process = await asyncio.create_subprocess_exec(*loc, env=env)
        await process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, loc)

    async def run_acoustic_inference(speaker, output_dir):
        output_file = f'{output_dir}/{file_name}.wav'
        if not os.path.exists(output_file) or overwrite == 1:
            loc = ['python', 'scripts/infer.py', 'acoustic', f'{root_path}/data/var_ds/{test_name}/{file_name}.ds', '--exp', 'muer_singer', '--spk', speaker, '--out', output_dir, '--batch_size', str(batch_size)]
            await run_subprocess(loc)

    async def run_variance_inference(output_dir):
        output_file = f'{output_dir}/{file_name}.ds'
        if not os.path.exists(output_file) or overwrite == 1:
            loc = ['python', 'onnxrun/infer_onnx.py', 'variance', f'{root_path}/data/ds/{test_name}/{file_name}.ds', '--exp', 'cpop_variance', '--out', output_dir, '--predict', 'pitch', '--predict', 'dur', '--batch_size', str(batch_size)]
            await run_subprocess(loc)
    # 先运行 run_variance_inference
    await run_variance_inference(f'{root_path}/data/var_ds/{test_name}/')

    # 然后根据条件运行 run_acoustic_inference
    if 'cpop_female' in spk:
        await run_acoustic_inference('cpop_female', f'{root_path}/data/vocal/{test_name}/cpop_female/')

    if 'cpop_male' in spk:
        await run_acoustic_inference('cpop_male', f'{root_path}/data/vocal/{test_name}/cpop_male/')
    # tasks = [
    # run_variance_inference(f'{root_path}/data/var_ds/{test_name}/')
    # ]
    # acoustic_tasks = []
    # if 'cpop_female' in spk:
    #     acoustic_tasks.append(run_acoustic_inference('cpop_female', f'{root_path}/data/vocal/{test_name}/cpop_female/'))

    # if 'cpop_male' in spk:
    #     acoustic_tasks.append(run_acoustic_inference('cpop_male', f'{root_path}/data/vocal/{test_name}/cpop_male/'))

    # tasks.extend(acoustic_tasks)

    # await asyncio.gather(*tasks)
    # 设置环境变量  
    # env = os.environ.copy()  
    # env['CUDA_VISIBLE_DEVICES'] = str(CUDA)

    # output_file = f'{root_path}/data/var_ds/{test_name}/{file_name}.ds'
    # if not os.path.exists(output_file) or overwrite == 1:
    #     # loc = f'variance {root_path}/data/ds/{test_name}/{file_name}.ds --exp cpop_variance --out {root_path}/data/var_ds/{test_name}/ --predict pitch --predict dur --batch_size {batch_size}'
    #     loc = ['python', 'onnxrun/infer_onnx.py', 'variance', f'{root_path}/data/ds/{test_name}/{file_name}.ds', '--exp', 'cpop_variance', '--out', f'{root_path}/data/var_ds/{test_name}/', '--predict', 'pitch', '--predict', 'dur', '--batch_size', str(batch_size)]
    #     # completed_process = subprocess.run(loc, env=env, check=True)
    #     # assert completed_process.returncode == 0, f'Error occured when processing song {file_name}'
    #     await run_subprocess(loc)

    # output_file = f'{root_path}/data/vocal/{test_name}/cpop_female//{file_name}.wav'
    # if 'cpop_female' in spk:
    #     if not os.path.exists(output_file) or overwrite == 1:
    #         loc = ['python', 'scripts/infer.py', 'acoustic', f'{root_path}/data/var_ds/{test_name}/{file_name}.ds', '--exp', 'muer_singer', '--spk', 'cpop_female', '--out', f'{root_path}/data/vocal/{test_name}/cpop_female/', '--batch_size', str(batch_size)]
    #         # completed_process = subprocess.run(loc, env=env, check=True)
    #         # assert completed_process.returncode == 0, f'Error occurred when processing song {file_name}'
    #         await run_subprocess(loc)

    # output_file = f'{root_path}/data/vocal/{test_name}/cpop_male//{file_name}.wav'
    # if 'cpop_male' in spk:
    #     if not os.path.exists(output_file) or overwrite == 1:
    #         loc = ['python', 'scripts/infer.py', 'acoustic', f'{root_path}/data/var_ds/{test_name}/{file_name}.ds', '--exp', 'muer_singer', '--spk', 'cpop_male', '--out', f'{root_path}/data/vocal/{test_name}/cpop_male/', '--batch_size', str(batch_size)]
    #         # completed_process = subprocess.run(loc, env=env, check=True)
    #         # assert completed_process.returncode == 0, f'Error occurred when processing song {file_name}'
    #         await run_subprocess(loc)

async def generate_vocal(test_input):
    error_song_ids = []
    for song_id in song_ids[test_input.test_name]:
        try:
            await run_diffsinger(str(song_id), test_input.test_midi_file, test_input.spk, test_input.batch_size , test_input.CUDA, test_input.overwrite)
        except:
            error_song_ids.append(song_id)
    
    if len(error_song_ids) > 0:
        raise ValueError(f'Total {len(error_song_ids)} songs failed: {error_song_ids}')

def run_diffsinger_command(file_name, test_name, spk, batch_size=1, CUDA=7, overwrite=0):
    # 设置环境变量  
    env = os.environ.copy()  
    env['CUDA_VISIBLE_DEVICES'] = str(CUDA)

    output_file = f'{root_path}/data/var_ds/{test_name}/{file_name}.ds'
    if not os.path.exists(output_file) or overwrite == 1:
        loc = ['python', 'onnxrun/infer_onnx.py', 'variance', f'{root_path}/data/ds/{test_name}/{file_name}.ds', '--exp', 'cpop_variance', '--out', f'{root_path}/data/var_ds/{test_name}/', '--predict', 'pitch', '--predict', 'dur', '--batch_size', str(batch_size)]
        completed_process = subprocess.run(loc, env=env, check=True)
        assert completed_process.returncode == 0, f'Error occured when processing song {file_name}'

    output_file = f'{root_path}/data/vocal/{test_name}/cpop_female//{file_name}.wav'
    if 'cpop_female' in spk:
        if not os.path.exists(output_file) or overwrite == 1:
            loc = ['python', 'scripts/infer.py', 'acoustic', f'{root_path}/data/var_ds/{test_name}/{file_name}.ds', '--exp', 'muer_singer', '--spk', 'cpop_female', '--out', f'{root_path}/data/vocal/{test_name}/cpop_female/', '--batch_size', str(batch_size)]
            completed_process = subprocess.run(loc, env=env, check=True)
            assert completed_process.returncode == 0, f'Error occurred when processing song {file_name}'

    output_file = f'{root_path}/data/vocal/{test_name}/cpop_male//{file_name}.wav'
    if 'cpop_male' in spk:
        if not os.path.exists(output_file) or overwrite == 1:
            loc = ['python', 'scripts/infer.py', 'acoustic', f'{root_path}/data/var_ds/{test_name}/{file_name}.ds', '--exp', 'muer_singer', '--spk', 'cpop_male', '--out', f'{root_path}/data/vocal/{test_name}/cpop_male/', '--batch_size', str(batch_size)]
            completed_process = subprocess.run(loc, env=env, check=True)
            assert completed_process.returncode == 0, f'Error occurred when processing song {file_name}'

if __name__ == '__main__':
    # 设置根目录
    root_dir = "/export/data/home/john/MuerSinger2/DiffSinger"
    # 更改工作目录到根目录
    os.chdir(root_dir)

    # 创建解析器
    parser = argparse.ArgumentParser()
    # 添加参数
    parser.add_argument('--test_name', type=str, default='test1', help='测试名称')
    parser.add_argument('--test_midi_file', type=str, default='test1', help='测试 MIDI 文件路径')
    parser.add_argument('--spk', type=list, default=['cpop_female'], help='歌手列表')
    parser.add_argument('--batch_size', type=int, default=1, help='CUDA 设备编号')
    parser.add_argument('--CUDA', type=int, default=7, help='CUDA 设备编号')
    parser.add_argument('--overwrite', type=int, default=0, help='是否覆盖已有文件')
    # 解析命令行参数
    args = parser.parse_args()
    
    error_song_ids = []
    for song_id in song_ids[args.test_name]:
        try:
            run_diffsinger_command(str(song_id), args.test_midi_file, args.spk, args.batch_size , args.CUDA, args.overwrite)
        except:
            error_song_ids.append(song_id)
    
    if len(error_song_ids) > 0:
        logging.error(f"generate_vocal for test {args.test_midi_file} failed, error num {len(error_song_ids)}, error: {error_song_ids}") 
    else:
        logging.info(f"generate_vocal for test {args.test_midi_file} successful")