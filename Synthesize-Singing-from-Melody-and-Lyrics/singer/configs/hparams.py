# Copyright (c) 2023 MusicBeing Project. All Rights Reserved.
#
# Author: Tao Zhang <ztao8@hotmail.com>
# Revised by: Yongsheng Feng
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

import os
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from singer.utils import logging
from singer.utils import probe_devices

""" Global hyper-parameters for the project.

This module defines global hyper-parameters for the project. It is used by
both training and inference.

The hyper-parameters are fetched in the following order:

    1. From command line arguments.
    2. From environment variables.
    3. From the "cfg" section in the model file.
    4. From the "config.yaml" file in the "configs" directory in each model.
    5. From the "singer-config.yaml" file in the "configs" directory in the project.
"""

logger = logging.get_logger(__name__)


#######################################################################################################
# Global singleton instance of configurations.
#######################################################################################################
_configs = OmegaConf.create()


#######################################################################################################
# Functions for hyper-parameters.
#######################################################################################################
def init_hparams(config_file="singer-config.yaml"):
    """Initialize hyper-parameters."""
    global _configs
    _configs = OmegaConf.create()

    # Load hyper-parameters from the "singer-config.yaml" file in the "configs" directory in the project.
    root_path = Path(__file__).parent.parent.parent
    config_path = root_path.joinpath("configs").joinpath(config_file)
    logger.info(f"Loading hyper-parameters from {config_path} ...")

    if config_path.exists():
        _configs = OmegaConf.load(config_path)
        assert isinstance(_configs, DictConfig)

    else:
        logger.error(f"Hyper-parameters not found in {config_path}.")
        _configs = OmegaConf.create()

    # Add the "singer" section if it does not exist.
    if "singer" not in _configs:
        _configs.singer = OmegaConf.create()

    _configs.singer.root_path = root_path

    # Load hyper-parameters from the "config.yaml" file in the "configs" directory in each model.
    if "models" not in _configs.singer:
        _configs.singer.models = OmegaConf.create()

    models_path = root_path.joinpath("singer").joinpath("models")

    for model_path in models_path.iterdir():
        if model_path.is_dir():
            model_name = model_path.name
            model_config_path = model_path.joinpath("configs").joinpath("config.yaml")
            if model_config_path.exists():
                logger.info(f"Loading hyper-parameters for the '{model_name}' model, from {model_config_path} ...")
                model_configs = OmegaConf.load(model_config_path)
                assert isinstance(model_configs, DictConfig)
                _configs.singer.models = OmegaConf.merge(_configs.singer.models, model_configs)

    trainers_cfg_dir = root_path.joinpath("singer").joinpath("training", "configs")
    if trainers_cfg_dir.exists() and trainers_cfg_dir.is_dir():
        trainer_config_file = trainers_cfg_dir.joinpath("config.yaml")
        if trainer_config_file.exists():
            logger.info(f"Loading hyper-parameters for training, from {trainer_config_file} ...")
            trainer_configs = OmegaConf.load(trainer_config_file)
            assert isinstance(trainer_configs, DictConfig)
            _configs.singer.trainers = trainer_configs


def get_hparams() -> dict:
    """Get hyper-parameters."""
    return OmegaConf.to_container(_configs, resolve=True)


def merge_hparams(hparams: DictConfig or dict):
    """Merge hyper-parameters."""
    global _configs
    _configs = OmegaConf.merge(_configs, hparams)


def _merge_hparams_with_env():
    """Merge hyper-parameters from environment variables."""
    global _configs

    _configs.env = OmegaConf.create()

    for key, value in os.environ.items():
        key = key.upper()
        _configs.env[key] = value


def _merge_hparams_with_cli():
    """Merge hyper-parameters from command line arguments."""
    global _configs

    _configs.cli = OmegaConf.from_cli()


def post_init_hparams():
    """Should be called after the hyper-paramers are initialized, and
    before they are used.
    """
    _merge_hparams_with_env()
    _merge_hparams_with_cli()

    # Probe the device
    device, device_ids = probe_devices()
    if "device" not in _configs.singer:
        _configs.singer.device = device

    if "device_ids" not in _configs.singer:
        _configs.singer.device_ids = device_ids
