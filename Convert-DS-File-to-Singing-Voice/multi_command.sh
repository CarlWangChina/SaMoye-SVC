#!/bin/bash
# conda activate diffsinger ln -s /export/data/home/xary/projects/ /home/john/MuerSinger2/xary_proj/ chmod -R 777 /export/data/home/xary/projects
# [54, 55, 62, 63, 70, 71, 78, 79, 86, 87, 94]
    
    # "./singer600_midi_to_vocal.sh -m singer600_2_test11 -c 0"
    # "./singer600_midi_to_vocal.sh -m singer600_2_test12 -c 1"
    # "./singer600_midi_to_vocal.sh -m singer600_2_test13 -c 2"
    # "./singer600_midi_to_vocal.sh -m singer600_2_test14 -c 3"
    # "./singer600_midi_to_vocal.sh -m singer600_2_test15 -c 4"
    # "./singer600_midi_to_vocal.sh -m singer600_2_test16 -c 5"
    # "./singer600_midi_to_vocal.sh -m singer600_2_test17 -c 6"
    
    # "./singer600_midi_to_vocal.sh -m singer600_2_test18 -c 0"
    # "./singer600_midi_to_vocal.sh -m singer600_2_test19 -c 1"
    # "./singer600_midi_to_vocal.sh -m singer600_2_test20 -c 2"
    # "./singer600_midi_to_vocal.sh -m singer600_2_test21 -c 3"
    # "./singer600_midi_to_vocal.sh -m singer600_2_test22 -c 4"
    # "./singer600_midi_to_vocal.sh -m singer600_2_test23 -c 5"
    # "./singer600_midi_to_vocal.sh -m singer600_2_test24 -c 6"
# 定义要执行的命令列表
commands=(
    "./singer600_midi_to_vocal.sh -m singer600_2_test25 -c 1"
    "./singer600_midi_to_vocal.sh -m singer600_2_test26 -c 2"
    "./singer600_midi_to_vocal.sh -m singer600_2_test27 -c 3"
    "./singer600_midi_to_vocal.sh -m singer600_2_test28 -c 4"
    "./singer600_midi_to_vocal.sh -m singer600_2_test29 -c 5"
    "./singer600_midi_to_vocal.sh -m singer600_2_test30 -c 6"
    "./singer600_midi_to_vocal.sh -m singer600_2_test31 -c 7"

    "./singer600_midi_to_vocal.sh -m singer600_2_test32 -c 1"
    "./singer600_midi_to_vocal.sh -m singer600_2_test33 -c 2"
    "./singer600_midi_to_vocal.sh -m singer600_2_test34 -c 3"
    "./singer600_midi_to_vocal.sh -m singer600_2_test35 -c 4"
    "./singer600_midi_to_vocal.sh -m singer600_2_test36 -c 5"
    "./singer600_midi_to_vocal.sh -m singer600_2_test37 -c 6"
    "./singer600_midi_to_vocal.sh -m singer600_2_test38 -c 7"

    "./singer600_midi_to_vocal.sh -m singer600_2_test39 -c 1"
    "./singer600_midi_to_vocal.sh -m singer600_2_test40 -c 2"
    "./singer600_midi_to_vocal.sh -m singer600_2_test41 -c 3"
    "./singer600_midi_to_vocal.sh -m singer600_2_test42 -c 4"
    "./singer600_midi_to_vocal.sh -m singer600_2_test43 -c 5"
    "./singer600_midi_to_vocal.sh -m singer600_2_test44 -c 6"
    "./singer600_midi_to_vocal.sh -m singer600_2_test45 -c 7"
)


# 遍历命令列表
for command in "${commands[@]}"; do
    # 提取命令中的 test 名字
    test_name=$(echo "$command" | awk '{split($0, arr, " "); print arr[3]}')

    # 定义日志文件路径
    log_file="log/${test_name}_log.txt"

    # 执行命令并将输出重定向到日志文件中
    $command >> "$log_file" 2>&1 &

    echo "Command $command is running. Log file is $log_file"
done
