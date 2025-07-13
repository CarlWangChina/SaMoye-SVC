from omegaconf import OmegaConf

from .logger import get_logger

log = get_logger(__name__)


_hparams = {}


def get_hparams():
    global _hparams
    if _hparams == {}:
        config_file_path = "configs/base.yaml"
        _hparams = OmegaConf.to_container(OmegaConf.load(config_file_path))
        log.info(f"Loading config from {config_file_path}")
    return _hparams
