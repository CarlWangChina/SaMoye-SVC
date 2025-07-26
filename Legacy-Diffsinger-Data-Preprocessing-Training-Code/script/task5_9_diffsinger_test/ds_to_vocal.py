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

from singer.models import run_diffsinger_command_test


def main(ds_output_path, vocal_path):
    # 将ds文件转为vocal
    run_diffsinger_command_test(ds_output_path=ds_output_path, vocal_path=vocal_path,key=-13)
    logger.info(f"vocal文件夹路径: {vocal_path}")
    return vocal_path


if __name__ == "__main__":
    import argparse

    """ python script/task5_9_diffsinger_test/ds_to_vocal.py \
    --ds_output_path /app/data/samoye-exp/singer/models/New_DiffSinger/samples \
    --vocal_path /app/data/samoye-exp/data/mq/human/test/vocal

    python script/task5_9_diffsinger_test/ds_to_vocal.py \
    --ds_output_path /app/data/samoye-exp/singer/models/Acoustc_Diffsinger/samples/00_我多想说再见啊.ds \
    --vocal_path /app/data/samoye-exp/data/mq/human/test/vocal
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_output_path", type=str, required=True)
    parser.add_argument("--vocal_path", type=str, required=True)
    args = parser.parse_args()

    ds_output_path = Path(args.ds_output_path)
    vocal_path = Path(args.vocal_path)

    if ds_output_path.suffix == ".ds":
        main(ds_output_path, vocal_path)
    else:
        for i, file_path in enumerate(ds_output_path.glob("*.ds")):
            if file_path.is_file():
                logger.info(f"ds_output_path: {file_path}")
            try:
                main(file_path, vocal_path)
            except Exception as e:
                logger.error(f"Error: {e}")
            if i > 6:
                break
