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

from singer.ds_process.midi_2_ds import Midi2Ds
from singer.models.infer_diffsinger import run_diffsinger_command

"""
    将midi转为ds
"""

if __name__ == "__main__":
    new_midi_file_name = "4-24-xm"
    midi_path = root_path / "data/human_midi/" / new_midi_file_name
    lyric_path = root_path / "data/new_lyric/" / new_midi_file_name
    ds_path = root_path / "data/new_ds/" / new_midi_file_name
    var_ds_path = root_path / "data/var_ds/" / new_midi_file_name
    vocal_path = root_path / "data/vocal/" / new_midi_file_name
    dict_file = root_path / "singer/models/Acoustc_Diffsinger/dictionaries/opencpop-extension.txt"
    ds_path.mkdir(parents=True, exist_ok=True)
    for midi_file in midi_path.glob("*.mid"):
        
        print(f"Processing {midi_file}")
        lyric_file = lyric_path / (midi_file.stem + ".txt")
        ds_file = ds_path / (midi_file.stem + ".ds")
        
        converter = Midi2Ds(midi_file, lyric_file, dict_file)
        converter.write_ds(ds_file)
        
        ds_input_path = ds_file
        ds_output_path = var_ds_path / f"{midi_file.stem}.ds"
        run_diffsinger_command(
            ds_input_path,
            ds_output_path,
            vocal_path
        )