import sys

from loguru import logger

class SingletonLogger:
    """
    Singleton class for logger
    in order to avoid multiple handlers with different logging levels
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SingletonLogger, cls).__new__(cls)

            cls._instance.logger = logger
            cls._instance.logger.remove()

            handler = {
                "sink": sys.stdout,
                "level": args[0]
            }
            cls._instance.logger.add(**handler)

        return cls._instance


def setup_logger(log_level: str):
    return SingletonLogger(log_level)
