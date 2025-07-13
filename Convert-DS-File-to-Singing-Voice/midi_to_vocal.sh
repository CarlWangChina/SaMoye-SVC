#!/bin/bash

# 解析命令行参数
while [[ $# -gt 0 ]]
do
    key="$1"

    case $key in
        --test_name)
        test_name="$2"
        shift # past argument
        shift # past value
        ;;
        --test_midi_file)
        test_midi_file="$2"
        shift # past argument
        shift # past value
        ;;
        --batch_size)
        batch_size="$2"
        shift # past argument
        shift # past value
        ;;
        --CUDA)
        CUDA="$2"
        shift # past argument
        shift # past value
        ;;
        *)    # unknown option
        shift # past argument
        ;;
    esac
done

export CUDA_VISIBLE_DEVICES=$CUDA

# 运行python脚本

export PATH="/home/xary/anaconda3/envs/singer/bin:$PATH"

python midi_lyric_align.py --test_name "$test_name" --test_midi_file "$test_midi_file"

python generate_vocal.py --test_name "$test_name" --test_midi_file "$test_midi_file" --batch_size "$batch_size" --CUDA "$CUDA"
