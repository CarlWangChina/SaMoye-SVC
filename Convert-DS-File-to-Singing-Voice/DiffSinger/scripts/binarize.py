import importlib
import os
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.resolve()
os.environ['PYTHONPATH'] = str(root_dir)
sys.path.insert(0, str(root_dir))

from utils.hparams import set_hparams, hparams

set_hparams()


def binarize():
    binarizer_cls = hparams.get("binarizer_cls", 'basics.base_binarizer.BaseBinarizer')
    pkg = ".".join(binarizer_cls.split(".")[:-1])
    cls_name = binarizer_cls.split(".")[-1]
    binarizer_cls = getattr(importlib.import_module(pkg), cls_name)
    print("| Binarizer: ", binarizer_cls)
    binarizer_cls().process()


if __name__ == '__main__':
    binarize()
