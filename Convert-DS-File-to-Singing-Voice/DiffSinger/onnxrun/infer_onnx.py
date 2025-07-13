import json
import os
import pathlib
import sys
from collections import OrderedDict
from pathlib import Path
import time
import click
from typing import Tuple
import torch   


root_dir = Path(__file__).parent.parent.resolve()
os.environ['PYTHONPATH'] = str(root_dir)
sys.path.insert(0, str(root_dir))


def find_exp(exp):
    if not (root_dir / 'checkpoints' / exp).exists():
        for subdir in (root_dir / 'checkpoints').iterdir():
            if not subdir.is_dir():
                continue
            if subdir.name.startswith(exp):
                # print(f'| match ckpt by prefix: {subdir.name}')
                exp = subdir.name
                break
        else:
            assert False, \
                f'There are no matching exp starting with \'{exp}\' in \'checkpoints\' folder. ' \
                'Please specify \'--exp\' as the folder name or prefix.'
    else:
        print(f'| found ckpt by name: {exp}')
    return exp


@click.group()
def main():
    pass

@main.command(help='Run DiffSinger ONNX acoustic model inference')
@click.argument('proj', type=str, metavar='DS_FILE')
@click.option('--exp', type=str, required=True, metavar='EXP', help='Selection of model')
@click.option('--spk', type=str, required=False, help='Speaker name or mix of speakers')
@click.option('--out', type=str, required=False, metavar='DIR', help='Path of the output folder')
@click.option('--title', type=str, required=False, help='Title of output file')
@click.option('--num', type=int, required=False, default=1, help='Number of runs')
@click.option('--key', type=int, required=False, default=0, help='Key transition of pitch')
@click.option('--gender', type=float, required=False, help='Formant shifting (gender control)')
@click.option('--seed', type=int, required=False, default=-1, help='Random seed of the inference')
@click.option('--depth', type=int, required=False, default=-1, help='Shallow diffusion depth')
@click.option('--speedup', type=int, required=False, default=0, help='Diffusion acceleration ratio')
@click.option('--mel', is_flag=True, help='Save intermediate mel format instead of waveform')
@click.option('--batch_size', type=int, required=False, default=1, help='Diffusion acceleration ratio')
@click.option('--onnx_model', type=str, required=False, default='Llane_Crow_v125', help='onnx model name')
def acoustic(
        proj: str,
        exp: str,
        spk: str,
        out: str,
        title: str,
        num: int,
        key: int,
        gender: float,
        seed: int,
        depth: int,
        speedup: int,
        mel: bool,
        batch_size: int,
        onnx_model: str
):
    start_time = time.time()
    
    proj = pathlib.Path(proj).resolve()
    name = proj.stem if not title else title
    exp = find_exp(exp)
    if out:
        out = pathlib.Path(out)
    else:
        out = proj.parent

    if gender is not None:
        assert -1 <= gender <= 1, 'Gender must be in [-1, 1].'

    with open(proj, 'r', encoding='utf-8') as f:
        params = json.load(f)

    if not isinstance(params, list):
        params = [params]

    if len(params) == 0:
        print('The input file is empty.')
        exit()

    from utils.infer_utils import trans_key, parse_commandline_spk_mix

    if key != 0:
        params = trans_key(params, key)
        key_suffix = '%+dkey' % key
        if not title:
            name += key_suffix
        print(f'| key transition: {key:+d}')

    sys.argv = [
        sys.argv[0],
        '--exp_name',
        exp,
        '--infer'
    ]
    from utils.hparams import set_hparams, hparams
    set_hparams()

    # Check for vocoder path
    assert mel or (root_dir / hparams['vocoder_ckpt']).exists(), \
        f'Vocoder ckpt \'{hparams["vocoder_ckpt"]}\' not found. ' \
        f'Please put it to the checkpoints directory to run inference.'

    if depth >= 0:
        assert depth <= hparams['K_step'], f'Diffusion depth should not be larger than K_step {hparams["K_step"]}.'
        hparams['K_step_infer'] = depth
    elif hparams.get('use_shallow_diffusion', False):
        depth = hparams['K_step_infer']
    else:
        depth = hparams['K_step']  # gaussian start (full depth diffusion)

    if speedup > 0:
        assert depth % speedup == 0, f'Acceleration ratio must be factor of diffusion depth {depth}.'
        hparams['pndm_speedup'] = speedup
    hparams['batch_size'] = batch_size
    spk_mix = parse_commandline_spk_mix(spk) if hparams['use_spk_id'] and spk is not None else None
    for param in params:
        if gender is not None and hparams.get('use_key_shift_embed'):
            param['gender'] = gender

        if spk_mix is not None:
            param['spk_mix'] = spk_mix

    hparams['onnx_model_dir'] = pathlib.Path('onnxrun/onnx_model/') / onnx_model
    from ds_onnx_acoustic import DiffSingerONNXAcousticInfer
    infer_ins = DiffSingerONNXAcousticInfer(load_vocoder=not mel)
    # print(f'| Model: {type(infer_ins.model)}')

    try:
        infer_ins.run_inference(
            params, out_dir=out, title=name, num_runs=num,
            spk_mix=spk_mix, seed=seed, save_mel=mel
        )
    except KeyboardInterrupt:
        exit(-1)
    endtime = time.time()
    # print(f'| Total time cost: {endtime - start_time:.2f}s')


@main.command(help='Run DiffSinger ONNX variance model inference')
@click.argument('proj', type=str, metavar='DS_FILE')
@click.option('--exp', type=str, required=True, metavar='EXP', help='Selection of model')
@click.option('--predict', type=str, multiple=True, metavar='TAGS', help='Parameters to predict')
@click.option('--out', type=str, required=False, metavar='DIR', help='Path of the output folder')
@click.option('--title', type=str, required=False, help='Title of output file')
@click.option('--num', type=int, required=False, default=1, help='Number of runs')
@click.option('--seed', type=int, required=False, default=-1, help='Random seed of the inference')
@click.option('--speedup', type=int, required=False, default=0, help='Diffusion acceleration ratio')
@click.option('--batch_size', type=int, required=False, default=1, help='Diffusion acceleration ratio')
@click.option('--device', type=str, required=False, default='None', help='Device to run inference')
def variance(
        proj: str,
        exp: str,
        out: str,
        num: int,
        speedup: int,
        predict: Tuple[str],
        seed: int,
        title: str,
        batch_size: int,
        device: str
        ):
    start_time = time.time()
    
    proj = pathlib.Path(proj).resolve()
    name = proj.stem if not title else title
    if out:
        out = pathlib.Path(out)
    else:
        out = proj.parent
    if (not out or out.resolve() == proj.parent.resolve()) and not title:
        name += '_variance'

    with open(proj, 'r', encoding='utf-8') as f:
        params = json.load(f)

    if not isinstance(params, list):
        params = [params]
    params = [OrderedDict(p) for p in params]

    if len(params) == 0:
        print('The input file is empty.')
        exit()

    sys.argv = [
        sys.argv[0],
        '--exp_name',
        exp,
        '--infer'
    ]
    from utils.hparams import set_hparams, hparams
    set_hparams()
    hparams['hop_size'] = 512
    hparams['audio_sample_rate'] = 44100
    hparams['batch_size'] = batch_size
    from ds_onnx_variance import DiffSingerONNXVarianceInfer
    if device == 'cpu':
        infer_ins = DiffSingerONNXVarianceInfer(device=device, predictions=set(predict))
    else:
        infer_ins = DiffSingerONNXVarianceInfer(predictions=set(predict))
    # print(f'| Model: {type(infer_ins.model)}')
    

    try:
        infer_ins.run_inference(
            params, out_dir=out, title=name,
            num_runs=num, seed=seed
        )
    except KeyboardInterrupt:
        exit(-1)

    end_time = time.time()
    # 获取最大内存使用量
    # max_memory = torch.cuda.max_memory_allocated()
    # print(f"Max GPU memory usage: {max_memory / 1024 / 1024:.2f} MB")
    # process_time = end_time - start_time
    # time_tatio = (process_time) / infer_ins.total_time
    # print(f'Total Inference time: {process_time:.2f}s ,Total Song Time:{infer_ins.total_time:.2f}s,Total  Time ratio: {time_tatio:.2f}')
    # import csv
    # # Prepare the data
    # data = [proj, f'{process_time:.2f}', f'{infer_ins.total_time:.2f}', f'{time_tatio:.2f}']
    # # Write the data to a CSV file
    # with open('var_output.csv', 'a', newline='') as file:
    # # Append the data to the CSV file
    #     writer = csv.writer(file)
    #     writer.writerow(data)

if __name__ == '__main__':
    # python onnxrun/infer_onnx.py variance samples/02_一半一半.ds --exp cpop_variance --out output/infer_onnx_out --predict pitch
    
    main()
    
    # 流程梳理：
    # 1. 通过main()函数，调用variance()函数
    # 2. variance()函数中，调用 DiffSingerONNXVarianceInfer()类，实例化infer_ins对象
    # 3. DiffSingerONNXVarianceInfer()类中，调用run_inference()函数
    # 4. run_inference()函数中，调用DsPitch 类的 Process 函数
