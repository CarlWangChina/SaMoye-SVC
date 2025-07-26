############################## 添加母目录到路径中 ###########################################
import sys
import os
from pathlib import Path

# 写log的Success Error 到 root_path / "log" / "ds_change_midi_to_vocal.log"
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))

from singer.utils.logging import get_logger_save_log

logger = get_logger_save_log(__name__, root_path / "log" / "fast_api_app.log")

###########################################################################################

from fastapi import FastAPI, HTTPException, Body
import uvicorn
from singer.ds_process.utils import extract_longest_digits_as_song_id
from singer.origin_process import new_midi_to_vocal
from singer.ds_process.utils import get_spk

app = FastAPI()
# uvicorn server:app --reload --host 127.0.1.1 --port 8000 --workers 16
# curl -X POST "http://127.0.1.1:8000/new_midi_to_vocal/" -d "new_midi_file_path='/app/data/midi_new/4-23-new-midi/_BALMY3_129191_revision_1.mid'&vocal_file_path='/app/data/vocal'"


@app.post("/new_midi_to_vocal/")
def fast_api_new_midi_to_vocal(
    new_midi_file_path: str = Body(
        ..., embed=True, description="Path to the new MIDI file"
    ),
    vocal_file_path: str = Body(
        ..., embed=True, description="Path to save the generated vocal file"
    ),
):
    """
    读取ds文件,将note转为新的，然后生成新的vocal
    """
    # 运行目录改为根目录
    os.chdir(root_path)

    path_new_midi_file_path = Path(str(new_midi_file_path))
    path_vocal_file_path = Path(str(vocal_file_path))

    song_id_dir_name = path_new_midi_file_path.parent.name
    song_id = extract_longest_digits_as_song_id(song_id_dir_name)
    if not song_id:
        raise HTTPException(
            status_code=400, detail="Could not extract song_id from new_midi_file_path"
        )
    # 假设文件命名遵循一定的模式，这里是一个示例
    ds_file_path = (
        root_path / "data/quantum/ds/" / "500ALL_tempo" / f"{song_id}_quantized.ds"
    )

    # 如果vocal_file_path 没有文件名后缀，使用{song_id}_quantized.ds，有则使用其文件名
    if not path_vocal_file_path.suffix:
        new_ds_file_name = f"{song_id}_new_midi_quantized.ds"
        vocal_path = path_vocal_file_path
    else:
        new_ds_file_name = f"{path_vocal_file_path.stem}.ds"
        vocal_path = path_vocal_file_path.parent

    new_ds_file_path = (
        root_path
        / "data/quantum/new_midi_ds/"
        / "500ALL_tempo"
        / song_id_dir_name
        / f"{new_ds_file_name}"
    )
    var_ds_file_path = (
        root_path
        / "data/quantum/new_midi_var_ds/"
        / "500ALL_tempo"
        / song_id_dir_name
        / f"{new_ds_file_name}"
    )
    new_ds_file_path.parent.mkdir(parents=True, exist_ok=True)
    var_ds_file_path.parent.mkdir(parents=True, exist_ok=True)

    spk = get_spk(str(new_midi_file_path))
    try:
        new_midi_to_vocal(
            ds_file_path,
            new_midi_file_path,
            new_ds_file_path,
            var_ds_file_path,
            vocal_path,
            spk,
            "/app/bin/python",
        )
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}")


if __name__ == "__main__":
    # sudo /app/bin/python script/fast_api/server.py
    uvicorn.run(app="server:app", host="127.0.1.1", port=8000, workers=32)

