import os
from pydub import AudioSegment
import tqdm
def convert_to_32kbps(input_folder, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹下的所有文件
    for filename in tqdm.tqdm(os.listdir(input_folder)):
        if filename.endswith(".mp3"):
            # 构建输入文件和输出文件的路径
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 加载 MP3 文件
            audio = AudioSegment.from_mp3(input_path)

            # 设置比特率为 32 kbps
            audio = audio.set_frame_rate(32000)

            # 保存为新的 MP3 文件
            audio.export(output_path, format="mp3", bitrate="32k")
            # print(f"Converted {input_path} to {output_path}")

import argparse 
parser = argparse.ArgumentParser(description='Convert MP3 files to 32 kbps')
parser.add_argument('--folder_name', type=str, help='Input folder containing MP3 files')

args = parser.parse_args()
# 指定输入文件夹和输出文件夹
input_folder = f"/home/john/MuerSinger2/data/mix/{args.folder_name}/cpop_female"
output_folder = f"/home/john/MuerSinger2/data/mix/{args.folder_name}/cpop_female"

# 命令
# python src/mp3_2_32k.py --folder_name singer600_test3 && python src/mp3_2_32k.py --folder_name singer600_test4 && python src/mp3_2_32k.py --folder_name singer600_test5
# python src/mp3_2_32k.py --folder_name singer600_test6 && python src/mp3_2_32k.py --folder_name singer600_test7 && python src/mp3_2_32k.py --folder_name singer600_test8
# python src/mp3_2_32k.py --folder_name singer600_test9 && python src/mp3_2_32k.py --folder_name singer600_test10 && python src/mp3_2_32k.py --folder_name singer600_test11
# python src/mp3_2_32k.py --folder_name singer600_test12 && python src/mp3_2_32k.py --folder_name singer600_test13 && python src/mp3_2_32k.py --folder_name singer600_test14
# python src/mp3_2_32k.py --folder_name singer600_test15 && python src/mp3_2_32k.py --folder_name singer600_test16 && python src/mp3_2_32k.py --folder_name singer600_test17
# python src/mp3_2_32k.py --folder_name singer600_test18 && python src/mp3_2_32k.py --folder_name singer600_test19 && python src/mp3_2_32k.py --folder_name singer600_test20
# python src/mp3_2_32k.py --folder_name singer600_test21 && python src/mp3_2_32k.py --folder_name singer600_test22 && python src/mp3_2_32k.py --folder_name singer600_test23
# python src/mp3_2_32k.py --folder_name singer600_test24 && python src/mp3_2_32k.py --folder_name singer600_test25 && python src/mp3_2_32k.py --folder_name singer600_test49
# 执行转换
convert_to_32kbps(input_folder, output_folder)
