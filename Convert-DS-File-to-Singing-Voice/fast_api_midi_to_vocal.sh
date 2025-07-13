#!/bin/bash

# 默认参数值
TEST_NAME="singer600"
TEST_MIDI_FILE="singer600_test1"
SPK="[\"cpop_female\"]"
BATCH_SIZE="100"
CUDA="7"
MODIFIED="1"
FASTAPI="8001"
OVERWRITE="0"

# 解析命令行参数
while getopts ":t:m:s:b:c:f:" opt; do
  case $opt in
    t)
      TEST_NAME=$OPTARG
      ;;
    m)
      TEST_MIDI_FILE=$OPTARG
      ;;
    s)
      SPK=$OPTARG
      ;;
    b)
      BATCH_SIZE=$OPTARG
      ;;
    c)
      CUDA=$OPTARG
      ;;
    f)
      FASTAPI=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

song_ids=$(/home/john/miniconda3/envs/diffsinger/bin/python -c "from src.params import song_ids; print(' '.join(str(x) for x in song_ids['$TEST_NAME']))")
# echo "$TEST_NAME 对应的内容是：$song_ids"

# 第一、二步运行一次即可
# # 发送第一个curl请求：先获取sheetsage的ds,该ds位于data/ds/$TEST_NAME/目录下
# curl -X "POST" \
#   "http://127.0.1.1:"$FASTAPI"/get_ds/" \
#   -H "Content-Type: application/json" \
#   -d '{
#   "test_name": "'"$TEST_NAME"'",
#   "test_midi_file": "'"$TEST_NAME"'",
#   "spk": '"$SPK"',
#   "batch_size": '"$BATCH_SIZE"',
#   "CUDA": '"$CUDA"',
#   "modified": '"$MODIFIED"'
# }'
# echo ""
# exit

# # 发送第二个curl请求：根据ds获取新的midi模版，该midi模版位于data/midi_for_lyric/$TEST_NAME/目录下
# curl -X "POST" \
#   "http://127.0.1.1:"$FASTAPI"/convert_ds_to_midi/" \
#   -H "Content-Type: application/json" \
#   -d '{
#   "test_name": "'"$TEST_NAME"'",
#   "test_midi_file": "'"$TEST_NAME"'",
#   "spk": '"$SPK"',
#   "batch_size": '"$BATCH_SIZE"',
#   "CUDA": '"$CUDA"',
#   "modified": '"$MODIFIED"'
# }'

# echo ""
# exit

# 创建目录
# if [ ! -d "data/midi/$TEST_MIDI_FILE" ]; then
#     # 如果目录不存在，则创建目录
#     mkdir -p "data/midi/$TEST_MIDI_FILE"
# fi
# sudo chmod -R 777 "data/midi/$TEST_MIDI_FILE"

# # 大批量运行时使用，确保已经有新midi的不再重复运行：发送第四个curl请求：获取新的ds，该ds位于data/ds/$TEST_MIDI_FILE/目录下
# # 如果"data/midi/$TEST_MIDI_FILE"目录为空，则跳过发送第四个curl请求
# if [ "$(ls -A data/midi/$TEST_MIDI_FILE)" ]; then
#   echo "data/midi/$TEST_MIDI_FILE 目录不为空"
#   curl -X "POST" \
#     "http://127.0.1.1:"$FASTAPI"/change_ds_midi/" \
#     -H "Content-Type: application/json" \
#     -d '{
#     "test_name": "'"$TEST_NAME"'",
#     "test_midi_file": "'"$TEST_MIDI_FILE"'",
#     "spk": '"$SPK"',
#     "batch_size": '"$BATCH_SIZE"',
#     "CUDA": '"$CUDA"',
#     "modified": '"$MODIFIED"',
#     "overwrite": '"$OVERWRITE"'
#   }'
# fi

# for song_id in $song_ids; do
#   # 如果OVERWRITE为0，则跳过已经存在的文件
#   if [ $OVERWRITE -eq 0 ]; then
#     if [ -f "/export/data/home/john/MuerSinger2/data/ds/$TEST_MIDI_FILE/"$song_id".ds" ]; then
#       # echo "data/midi/$TEST_MIDI_FILE/"$song_id"_new.mid 已经存在，跳过"
#       continue
#     fi
#   fi
#   # 发送第三个curl请求：根据ds转换的midi生成新的midi，该midi位于data/midi/$TEST_MIDI_FILE/目录下
#   curl -X "POST" "http://127.0.0.0:"$FASTAPI"/generate-midi/" \
#    -H "accept: audio/midi"   -H "Content-Type: multipart/form-data"   \
#    -F "file=@/home/john/MuerSinger2/data/midi_for_lyric/$TEST_NAME/"$song_id".mid;type=audio/midi" \
#    -F "pitch_range=[57,70]" -F "device=cuda:$CUDA" \
#    --output "/export/data/home/john/MuerSinger2/data/midi/$TEST_MIDI_FILE/"$song_id"_new.mid"
# done

# echo ""
# # exit

# # 发送第四个curl请求：获取新的ds，该ds位于data/ds/$TEST_MIDI_FILE/目录下
# curl -X "POST" \
#   "http://127.0.1.1:"$FASTAPI"/change_ds_midi/" \
#   -H "Content-Type: application/json" \
#   -d '{
#   "test_name": "'"$TEST_NAME"'",
#   "test_midi_file": "'"$TEST_MIDI_FILE"'",
#   "spk": '"$SPK"',
#   "batch_size": '"$BATCH_SIZE"',
#   "CUDA": '"$CUDA"',
#   "modified": '"$MODIFIED"',
#   "overwrite": '"$OVERWRITE"'
# }'

# echo ""
# # exit

# # 发送第五个curl请求：生成人声
# curl -X "POST" \
#   "http://127.0.1.1:"$FASTAPI"/generate_vocal/" \
#   -H "Content-Type: application/json" \
#   -d '{
#   "test_name": "'"$TEST_NAME"'",
#   "test_midi_file": "'"$TEST_MIDI_FILE"'",
#   "spk": '"$SPK"',
#   "batch_size": '"$BATCH_SIZE"',
#   "CUDA": '"$CUDA"',
#   "modified": '"$MODIFIED"',
#   "overwrite": '"$OVERWRITE"'
# }'

# echo ""
# python src/wav_2_mp3.py --test_midi_file $TEST_MIDI_FILE
# exit


mp3_path="/export/data/home/john/MuerSinger2/data/origin_mp3/$TEST_NAME"
company_path="/export/data/home/john/MuerSinger2/separated/mdx_extra"

# # 设置CUDA_VISIBLE_DEVICES环境变量
export CUDA_VISIBLE_DEVICES=$CUDA

# 循环遍历song_ids列表
for song_id in $song_ids; do
    # 构建mp3_path和company_path
    mp3_file="$mp3_path/$song_id.mp3"
    company_file="$company_path/$song_id/no_vocals.wav"

    # 如果company_path不存在，则运行demucs命令
    if [ ! -f "$company_file" ]; then
        echo "company_path $company_file 不存在，运行demucs命令"
        demucs --two-stems vocals -n mdx_extra "$mp3_file"
    # else
    #     echo "company_path $company_file 存在，跳过"
    fi
done

# echo ""

# # 发送第六个curl请求：混音
# curl -X "POST" \
#   "http://127.0.1.1:"$FASTAPI"/mix_song/" \
#   -H "Content-Type: application/json" \
#   -d '{
#   "test_name": "'"$TEST_NAME"'",
#   "test_midi_file": "'"$TEST_MIDI_FILE"'",
#   "spk": '"$SPK"',
#   "batch_size": '"$BATCH_SIZE"',
#   "CUDA": '"$CUDA"',
#   "modified": '"$MODIFIED"'
# }'

# echo ""