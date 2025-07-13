from pydub import AudioSegment
import os
import tqdm
def convert_wav_to_mp3(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取输入文件夹中的所有wav文件
    wav_files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]

    # 使用tqpdm显示进度
    wav_files = tqdm.tqdm(wav_files)
    for wav_file in wav_files:
        input_path = os.path.join(input_folder, wav_file)
        output_file = os.path.splitext(wav_file)[0] + '.mp3'
        # mp3 文件已经存在，跳过
        if os.path.exists(os.path.join(output_folder, output_file)):
            continue

        # 读取wav文件
        audio = AudioSegment.from_wav(input_path)
        
        # 构建输出文件的路径
        
        output_path = os.path.join(output_folder, output_file)
        
        # 将音频文件转换为mp3格式并保存到输出文件夹
        audio.export(output_path, format="mp3")
        
        # print(f"Converted {wav_file} to {output_file}")

if __name__ == '__main__':
    import argparse
    # 定义参数
    parser = argparse.ArgumentParser(description='Convert wav files to mp3 files')
    parser.add_argument('--test_midi_file', type=str,default='singer600_test4', help='The folder containing wav files')
    args = parser.parse_args()
    # 指定输入和输出文件夹
    input_folder = f"/home/john/MuerSinger2/data/vocal/{args.test_midi_file}/cpop_female"
    output_folder = f"/home/john/MuerSinger2/data/vocal_mp3/{args.test_midi_file}/cpop_female"

    # 调用函数进行转换
    convert_wav_to_mp3(input_folder, output_folder)
