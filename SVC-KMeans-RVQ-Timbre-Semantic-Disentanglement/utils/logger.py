""" Define a logger for logging service informations.
"""

import logging
import logging.config
import os

# Make log save dir
logSaveDir = "logs"
os.makedirs(logSaveDir, exist_ok=True)

# 设置日志格式
loggingConfig = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s.%(msecs)03d | %(filename)s:%(lineno)s | %(levelname)s | pid=%(process)d | tid=%(thread)d | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    },
    "handlers": {
        "consoleHandler": {
            "class": "logging.StreamHandler",
            "formatter": "default",
        },
        "fileHandler": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "formatter": "default",
            "filename": f"{logSaveDir}/svc_service.log",
            "when": "D",
            "interval": 1,
            "backupCount": 7,
            "encoding": "utf8",
        }
    },
    "loggers": {
        "svc_logger": {
            "handlers": ["consoleHandler", "fileHandler"],
            "level": logging.INFO,
            "propagate": False,
        }
    }
}


def get_logger(name: str = None) -> logging.Logger:
    """ Define a svc logger.
    """
    if name is None:
        logger = logging.getLogger(name)
    else:
        logging.config.dictConfig(loggingConfig)
        logger = logging.getLogger("svc_logger")

    return logger
