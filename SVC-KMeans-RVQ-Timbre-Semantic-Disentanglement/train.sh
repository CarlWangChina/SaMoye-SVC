# /bin/bash

set -euo pipefail

python svc_trainer.py -c configs/zhangjian_read_20s.yaml -n sovits5.0
