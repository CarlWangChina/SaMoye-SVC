############################## 添加母目录到路径中 ###########################################
import sys
import os
from pathlib import Path

# 写log的Success Error 到 root_path / "log" / "ds_change_midi_to_vocal.log"
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))
###########################################################################################
from singer.mq import start as app_start


def main():
    app_start()


if __name__ == "__main__":
    main()
