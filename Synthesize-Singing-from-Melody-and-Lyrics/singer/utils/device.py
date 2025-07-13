# Copyright (c) 2023 MusicBeing Project. All Rights Reserved.
#
# Author: Tao Zhang <ztao8@hotmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from .logging import get_logger

logger = get_logger(__name__)


def probe_devices() -> (str, [int]):
    """Probe the available devices.

    Returns:
        device (str): The device available to use.
        device_ids [str]: A list of gpu devices available to use. If no gpu is available, the list is empty.
    """

    device = "cpu"
    device_ids = []

    logger.info("Probing devices...")
    if torch.cuda.is_available():
        device = "cuda"
        for i in range(torch.cuda.device_count()):
            device_ids.append(i)

    elif torch.backends.mps.is_available():
        device = "mps"

    logger.info(f"Device probed: {device}")

    return device, device_ids
