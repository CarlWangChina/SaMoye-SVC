""" convert wav to mp3"""

from pydub import AudioSegment


def convert_wav_to_mp3(input_path, output_path):
    """
    Convert wav to mp3

    Args:
        input_path (str): The path of the input wav file
        output_path (str): The path of the output mp3 file

    Returns:
        None

    """
    # 确保输出文件夹存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 读取wav文件
    audio = AudioSegment.from_wav(input_path)

    # 将音频文件转换为mp3格式并保存到输出文件夹
    audio.export(output_path, format="mp3", bitrate="64k")

    # print(f"Converted {wav_file} to {output_file}")
