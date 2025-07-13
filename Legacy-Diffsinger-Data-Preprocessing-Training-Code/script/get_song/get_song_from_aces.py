############################## 添加母目录到路径中 ###########################################
import sys
import os
from pathlib import Path

# 写log的Success Error 到 root_path / "log" / "ds_change_midi_to_vocal.log"
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))

from singer.utils.logging import get_logger_save_log

logger = get_logger_save_log(__name__, "log/get_song_from_aces.log")

###########################################################################################
from singer.ds_process.utils import extract_longest_digits_as_song_id
from singer.ace_process import ds_to_aces_to_vocal
from singer.ds_process.utils import get_spk


def change_new_midi_to_ace_vocal(
    new_midi_file_path: str,
    vocal_file_path: str,
    cuda: int = 0,
):
    """
    读取ds文件，将note转为新的，然后生成新的ace vocal
    """
    # 运行目录改为根目录
    os.chdir(root_path)

    path_new_midi_file_path = Path(str(new_midi_file_path))
    path_vocal_file_path = Path(str(vocal_file_path))

    song_id_dir_name = path_new_midi_file_path.parent.name  # songid 在目录名中
    # song_id_dir_name = path_new_midi_file_path.stem
    song_id = extract_longest_digits_as_song_id(song_id_dir_name)
    if not song_id:
        logger.error("Could not extract song_id from new_midi_file_path")
    ds_file_path = (
        root_path / "data/quantum/ds/" / "500ALL_tempo" / f"{song_id}_quantized.ds"
    )

    # 如果vocal_file_path 没有文件名后缀，使用{song_id}_quantized.ds，有则使用其文件名
    if not path_vocal_file_path.suffix:
        new_ds_file_name = ds_file_path.stem
        vocal_path = path_vocal_file_path
    else:
        new_ds_file_name = path_vocal_file_path.stem
        vocal_path = path_vocal_file_path.parent

    aces_path = root_path / "data/aces" / "500ALL_tempo" / song_id_dir_name / "aces"
    oggs_path = root_path / "data/aces" / "500ALL_tempo" / song_id_dir_name / "oggs"
    combine_audio_file_path = (
        root_path
        / "data/aces"
        / "500ALL_tempo"
        / song_id_dir_name
        / "combined"
        / f"{ds_file_path.stem}.wav"
    )
    combined_path = (
        root_path / "data/aces" / "500ALL_tempo" / song_id_dir_name / "combined"
    )
    # vocal_path = root_path / "data/aces" / "500ALL_tempo" / song_id_dir_name / "vocal"

    aces_path.mkdir(parents=True, exist_ok=True)
    oggs_path.mkdir(parents=True, exist_ok=True)
    combined_path.mkdir(parents=True, exist_ok=True)
    vocal_path.mkdir(parents=True, exist_ok=True)

    # 如果aces_path下有文件，删除
    for file in aces_path.iterdir():
        os.remove(file)
    # 如果oggs_path下有文件，删除
    for file in oggs_path.iterdir():
        os.remove(file)

    spk = get_spk(str(new_midi_file_path))
    # try:
    ds_to_aces_to_vocal(
        ds_file_path,
        aces_path,
        oggs_path,
        combine_audio_file_path,
        combined_path,
        vocal_path,
        new_ds_file_name,
        cuda,
        spk,
    )
    logger.info(f"Successfully converted {ds_file_path} to aces and vocal")
    # except Exception as e:
    #     logger.error(f"Failed to convert {ds_file_path} to aces and vocal")


if __name__ == "__main__":
    import argparse

    # python script/get_song/get_song_from_aces.py --new_midi_file_path /home/john/pipeline_for_6w/MuerSinger2/data/midi_new/4-23-new-midi/_BALMY3_129191_revision_1.mid --vocal_file_path /home/john/CaichongSinger/data/vocal/fast_api_test/_BALMY3_129191_revision_1.mp3 --cuda 7
    parser = argparse.ArgumentParser()
    parser.add_argument("--new_midi_file_path", type=str, required=True)
    parser.add_argument("--vocal_file_path", type=str, required=True)
    parser.add_argument("--cuda", type=int, default=0, help="cuda id, default 0")
    args = parser.parse_args()

    change_new_midi_to_ace_vocal(
        args.new_midi_file_path, args.vocal_file_path, args.cuda
    )
