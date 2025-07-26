############################## 添加母目录到路径中 ###########################################
import sys
import os
from pathlib import Path

# 写log的Success Error 到 root_path / "log" / "ds_change_midi_to_vocal.log"
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))

from singer.utils.logging import get_logger

logger = get_logger(__name__)

###########################################################################################

from singer.ace_process.aces_to_aces_api import aces_file_to_ogg

if __name__ == "__main__":
    aces_file_path = Path(
        "/app/data/temp/ace/98342e959a40497a936393e9a18c5e08/98342e959a40497a936393e9a18c5e08_sentence_2.aces"
    )
    ogg_path = Path("/app/data/temp/ogg/test_spk")

    speaker_ids = [0, 1, 2, 4, 5, 7, 18, 24, 26, 29, 46, 49, 76, 80, 81, 82, 83, 84]
    for speaker_id in speaker_ids:
        ogg_file_path = ogg_path / f"test_{speaker_id}.ogg"
        aces_file_to_ogg(aces_file_path, ogg_file_path, spk=str(speaker_id))
