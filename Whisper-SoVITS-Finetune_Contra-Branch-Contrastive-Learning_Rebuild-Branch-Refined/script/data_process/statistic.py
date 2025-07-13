# 两个目录/home/john/svc/muer-svc-whisper-sovits-svc/data/origin_data/Acapella /home/john/svc/muer-svc-whisper-sovits-svc/data/origin_data/open-source
# 1. 统计说话人数量 每个子文件夹是一个说话人
# 2. 统计每个说话人的音频数量及其音频总时长
import os
from pathlib import Path
from tqdm.contrib.concurrent import process_map
import librosa
import multiprocessing
import pandas as pd


def get_audio_info(audio_path):
    try:
        duration = librosa.get_duration(filename=audio_path)
        return duration
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return 0


def process_speaker(speaker_path):
    speaker_info = {}
    audio_files = list(speaker_path.glob("*.wav")) + list(speaker_path.glob("*.mp3"))
    num_audios = len(audio_files)
    total_duration = sum(process_map(get_audio_info, audio_files, max_workers=5))
    speaker_info["num_audios"] = num_audios
    speaker_info["total_duration"] = total_duration
    return speaker_info


def get_speaker_info(directory):
    speaker_info = {}
    speaker_dirs = [d for d in directory.iterdir() if d.is_dir()]
    results = process_map(process_speaker, speaker_dirs, max_workers=5)
    for speaker, result in zip(speaker_dirs, results):
        speaker_info[speaker.name] = result
    return speaker_info


def print_and_save_speaker_info(directory, description, output_file):
    print(f"\nStatistics for {description}:")
    directory_path = Path(directory)
    speaker_info = get_speaker_info(directory_path)
    total_speakers = len(speaker_info)
    print(f"Total number of speakers: {total_speakers}")

    data = []
    for speaker, info in speaker_info.items():
        print(
            f"Speaker: {speaker}, Number of audios: {info['num_audios']}, Total duration: {info['total_duration']:.2f} seconds, {info['total_duration']/3600:.2f} hours"
        )
        data.append(
            [
                speaker,
                info["num_audios"],
                info["total_duration"],
                info["total_duration"] / 3600,
            ]
        )

    # Save to CSV
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        df = df[df["Speaker"].str.contains(description) == False]
        data = data + df.values.tolist()
    df = pd.DataFrame(
        data,
        columns=[
            "Speaker",
            "Number of Audios",
            "Total Duration (seconds)",
            "Total Duration (hours)",
        ],
    )
    df.to_csv(output_file, index=False)
    print(f"\nSaved speaker information to {output_file}")


# Define the directories and output file
# acapella_dir = '/home/john/svc/muer-svc-whisper-sovits-svc/data/origin_data/Acapella'
# open_source_dir = '/home/john/svc/muer-svc-whisper-sovits-svc/data/origin_data/open-source'
# dir_1600 = "/home/john/svc/data/origin_data/New_1600"
damp_dir = "/home/john/svc/data/origin_data/New_DAMP"
output_file = "/home/john/svc/data/origin_data/speaker_info_damp.csv"

# Print and save statistics
# print_and_save_speaker_info(acapella_dir, "Acapella", output_file)
# print_and_save_speaker_info(open_source_dir, "Open Source", output_file)
# print_and_save_speaker_info(dir_1600, "1600", output_file)
print_and_save_speaker_info(damp_dir, "DAMP", output_file)
