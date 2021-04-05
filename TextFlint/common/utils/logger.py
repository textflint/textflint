import logging.config

LOG_STRING = "\033[34;1mTextFlint\033[0m"
logger = logging.getLogger(__name__)
logging.config.dictConfig(
    {"version": 1, "loggers": {__name__: {"level": logging.INFO}}}
)
formatter = logging.Formatter(f"{LOG_STRING}: %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.propagate = False
