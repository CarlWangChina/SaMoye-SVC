############################## 添加母目录到路径中 ###########################################
import sys
import os
from pathlib import Path
import logging

# 写log的Success Error 到 root_path / "log" / "ds_change_midi_to_vocal.log"

root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(root_path / "log" / "ds_change_midi_to_vocal.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

###########################################################################################

"""
    遍历新midi，找到对应的量化的ds文件，然后调用

"""
import argparse
import json
import re
import pretty_midi
from singer.ds_process.ds_change_note import ds_change_note
from singer.models.infer_diffsinger import run_diffsinger_command
from tqdm.contrib.concurrent import process_map

new_midi_file_name = "test_4_24_2"

songid_pattern = re.compile(r"_\d+")


def main(args):
    cuda, test_path = args
    new_midi_path = root_path / "data/new_midi/" / new_midi_file_name / test_path
    quantum_ds_path = root_path / "data/quantum_ds/4-23"
    new_ds_path = root_path / "data/new_ds/" / new_midi_file_name / test_path
    var_ds_path = root_path / "data/var_ds/" / new_midi_file_name / test_path
    vocal_path = root_path / "data/vocal/" / new_midi_file_name / test_path
    new_ds_path.mkdir(parents=True, exist_ok=True)
    var_ds_path.mkdir(parents=True, exist_ok=True)
    vocal_path.mkdir(parents=True, exist_ok=True)

    for midi_file in new_midi_path.glob("*.mid"):
        try:
            # _WAIT_433143_revision_1_female.mid 其中433143为songid female为spk参数
            # 获取文件名
            filename = midi_file.stem
            # 使用正则表达式查找songid
            match = songid_pattern.search(filename)
            if match:
                song_id = match.group(0)[1:]
                # song_id = match.group(0)
                logger.info(f"Find songid {song_id} in {filename}")
            else:
                logger.error(f"Cannot find songid in {filename}")
                continue

            revision = filename.split("_")[-2]
            if revision == "2":
                continue
            spk = filename.split("_")[-1]
            if spk == "male":
                spk = "cpop_male"
            elif spk == "female":
                spk = "cpop_female"
            else:
                logger.error(f"Cannot find spk in {filename}")
                continue
            logger.info(f"Processing {song_id}")

            ds_file = quantum_ds_path / f"{song_id}.ds"
            new_ds_file = new_ds_path / f"{filename}.ds"

            if not ds_file.exists():
                logger.error(f"{ds_file} not exists")
                continue

            with open(ds_file, "r") as f:
                origin_ds_data = json.load(f)
            midi_data = pretty_midi.PrettyMIDI(str(midi_file))

            # delete offset
            midi_offset = None
            melody_instrument = midi_data.instruments[2]
            if melody_instrument.is_drum is False and melody_instrument.program == 0:
                note_sorted = sorted(melody_instrument.notes, key=lambda x: x.start)
                midi_offset = note_sorted[0].start

            if midi_offset is not None:
                minus_offset = origin_ds_data[0]["offset"] - midi_offset + 0.3
                for i, sentence in enumerate(origin_ds_data):
                    origin_ds_data[i]["offset"] = (
                        float(sentence["offset"]) - minus_offset
                    )

            ds = ds_change_note(song_id, origin_ds_data, midi_data)

            # 如果ds 长于30s，则只保存前30s
            # offset_30 = float(ds[0]["offset"]) + 60
            # for s_i, sentence in enumerate(ds):
            #     if float(sentence["offset"]) >= offset_30:
            #         ds = ds[:s_i]
            #         break
            with open(new_ds_file, "w") as f:
                json.dump(ds, f, indent=4, ensure_ascii=False)

            # 生成新的vocal
            ds_input_path = new_ds_file
            ds_output_path = var_ds_path / f"{filename}.ds"
            vocal_path = vocal_path

            # 如果vocal已经存在，则跳过
            vocal_wav_path = vocal_path / f"{filename}.mp3"
            if vocal_wav_path.exists():
                logger.warning(f"{vocal_wav_path} exists")
                continue
            run_diffsinger_command(
                ds_input_path,
                ds_output_path,
                vocal_path,
                spk=spk,
                cuda=cuda,
            )
            logger.warning(f"Finish {midi_file}")
        except Exception as e:
            logger.error(f"Error in {midi_file}: {e}")
            continue


# 使用tqdm的process_map进行多线程处理

if __name__ == "__main__":
    # python script/task4_24_ds_change_midi_to_vocal/ds_change_midi_to_vocal.py --min 0 --max 256
    # find . -maxdepth 1 -mindepth 1 ! -name data -exec cp -r {} /nfs/john \;

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gpus",
        default="0,1,2,3,4,5,6,7",
        type=str,
        help="The GPU ids, comma-separated",
    )
    parser.add_argument("--min", default=0, type=int, help="The minimum index")
    parser.add_argument("--max", default=-1, type=int, help="The maximum index")

    args = parser.parse_args()

    gpus = [f"{gpu_id}" for gpu_id in map(int, args.gpus.split(","))]

    new_midi_path = root_path / "data/new_midi/" / new_midi_file_name
    new_midi_path_list = os.listdir(new_midi_path)

    # 为每个 new_midi_path 分配一个 gpu_id
    if args.max == -1:
        args.max = len(new_midi_path_list)
    zipped_args = list(
        zip(
            gpus * (len(new_midi_path_list) // len(gpus) + 1),
            new_midi_path_list[args.min : args.max],
        )
    )

    # 使用 process_map 来并行处理
    process_map(main, zipped_args, max_workers=len(gpus) * 4)

    #
