from pathlib import Path

from .ds_to_aces import ds_to_aces
from .aces_to_aces_api import process_aces_to_audio
from .merge_ogg import combine_audio
from .change_vocal2ds import vocal_to_ds
from .ds_2_vocal import split_ds_to_vocal


def postprocess_main(song_id, root_path: Path, ds_temp_path: Path = None):
    origin_ds_file = (
        root_path / Path(f"data/ds/temp/ace_input/{song_id}.ds")
        if ds_temp_path is None
        else ds_temp_path
    )
    ace_path = root_path / Path(f"data/aces/temp/{song_id}")
    ace_path.mkdir(parents=True, exist_ok=True)
    # 清空ace_path 上的aces文件
    for file in ace_path.iterdir():
        file.unlink()
    ds_to_aces(origin_ds_file, ace_path)
    audio_path = root_path / Path(f"data/audio_ace/temp/{song_id}")
    audio_path.mkdir(parents=True, exist_ok=True)
    # 清空audio_path 上的 ogg 音频文件
    for file in audio_path.iterdir():
        file.unlink()
    process_aces_to_audio(ace_path, audio_path)
    combined_path = root_path / Path(f"data/combined/temp/{song_id}")
    combined_path.mkdir(parents=True, exist_ok=True)
    combine_audio(song_id, audio_path, ace_path, combined_path)

    vocal_to_ds(origin_ds_file, combined_path, cuda=0)
    ds_path = root_path / Path(f"data/combined/temp/{song_id}/{song_id}.ds")
    new_ds_path = root_path / Path(f"data/combined/temp/{song_id}/{song_id}_split.ds")
    vocal_path = root_path / Path(f"data/vocal/{song_id}")
    # vocal_path = root_path / Path(f"data/vocal/temp") # 正式流程用这个
    vocal_path.mkdir(parents=True, exist_ok=True)
    split_ds_to_vocal(ds_path, new_ds_path, vocal_path)
    return vocal_path

def convert_vocal2ds2vocal(origin_ds_path, root_path: Path, song_id):
    convert_path = root_path / Path(f"data/combined/convert/{song_id}")
    vocal_to_ds(origin_ds_path, convert_path, cuda=0)
    ds_path = root_path / Path(f"data/combined/convert/{song_id}/{song_id}.ds")
    new_ds_path = root_path / Path(f"data/combined/convert/{song_id}/{song_id}_split.ds")
    vocal_path = root_path / Path(f"data/vocal/convert/{song_id}")
    vocal_path.mkdir(parents=True, exist_ok=True)
    # vocal_path = root_path / Path(f"data/vocal/temp") # 正式流程用这个
    split_ds_to_vocal(ds_path, new_ds_path, vocal_path)

# if __name__ == "__main__":
#     song_id = "1686672"
#     root_path = Path("/home/john/DuiniutanqinSinger")
    
#     # postprocess_main(song_id, root_path)
#     convert_vocal2ds2vocal(root_path / Path(f"/home/john/DuiniutanqinSinger/data/ds/temp/{song_id}.ds"), root_path, song_id)