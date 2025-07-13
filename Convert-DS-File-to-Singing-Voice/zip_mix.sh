#!/bin/bash

# 数据目录
data_dir="data/mix/"
# 目标目录
target_dir="/export/data/demo"

# 找到所有符合条件的子目录并进行压缩
find "$data_dir" -mindepth 1 -maxdepth 1 -type d -name 'singer600_test*' | while read -r dir; do
    # 提取子目录名称
    dir_name=$(basename "$dir")
    # 目标 zip 文件路径
    zip_file="$target_dir/${dir_name}.zip"
    
    # 如果目标 zip 文件已存在，则跳过
    if [ -f "$zip_file" ]; then
        echo "Zip file $zip_file already exists. Skipping..."
    else
        # 压缩子目录到目标目录
        zip -r "$zip_file" "$dir"
        echo "Zip file $zip_file created."
    fi
done
