import sys

from loguru import logger

class SingletonLogger:
    """
    Singleton class for logger
    in order to avoid multiple handlers with different logging levels
    """
    _instance = None
    _initialized = False
    logger = logger

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SingletonLogger, cls).__new__(cls)

        return cls._instance

    def setup_log_level(self, *args):
        if not self._initialized:
            self.logger.remove()

            handler = {
                "sink": sys.stdout,
                "level": args[0]
            }
            self.logger.add(**handler)
            self._initialized = True

    def debug(self, *args, **kwargs):
        self.logger.debug(*args, **kwargs)

    def info(self, *args, **kwargs):
        self.logger.info(*args, **kwargs)

    def warning(self, *args, **kwargs):
        self.logger.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        self.logger.error(*args, **kwargs)

    def critical(self, *args, **kwargs):
        self.logger.critical(*args, **kwargs)


custom_logger = SingletonLogger()