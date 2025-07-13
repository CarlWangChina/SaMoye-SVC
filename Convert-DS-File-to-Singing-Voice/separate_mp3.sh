#!/bin/bash

mp3_path="/export/data/home/john/MuerSinger2/data/origin_mp3/singer600"
song_ids=(523572 903269 1169589 1686672 153343 973273 847575 1303121 888509 880101 122844 1287594)
mp3_paths=()

# 构建mp3_paths列表
for song_id in "${song_ids[@]}"; do
    mp3_paths+=("$mp3_path/$song_id.mp3")
done

# 设置CUDA_VISIBLE_DEVICES环境变量
export CUDA_VISIBLE_DEVICES=0

# 使用demucs命令处理每个MP3文件
for mp3_path in "${mp3_paths[@]}"; do
    demucs --two-stems vocals -n mdx_extra "$mp3_path"
done
