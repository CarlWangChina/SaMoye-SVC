# Copyright (c) 2024 MusicBeing Project. All Rights Reserved.
#
# Author: Feee <cgoxopx@outlook.com>
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from auenhan.separation import *
from auenhan.config_loader import *

if __name__ == "__main__":
    processor = DemuxExtractor(use_denoise=config.processor.use_denoise)
    processor.process_file(infile=PROJECT_ROOT+"/tests/data/530adda3-e057-4c00-846f-1d8b8ab67eb3.mp3",
                           outfile_acc=PROJECT_ROOT+"/tests/data/530adda3-e057-4c00-846f-1d8b8ab67eb3_acc.mp3",
                           outfile_vocal=PROJECT_ROOT+"/tests/data/530adda3-e057-4c00-846f-1d8b8ab67eb3_vocal.mp3")
