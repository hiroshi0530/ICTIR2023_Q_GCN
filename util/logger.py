import sys
from loguru import logger


def init_logger(config):
    logger.remove()

    logger.add(
        sys.stdout,
        level=config["log_level"],
        colorize=True,
        format="<level>{time:YYYY-MM-DDTHH:mm:ss} {extra[env]} {level} [f:{function}] [L:{line}] {message}</level>",
    )

    logger.add(
        "./log/debug",
        level=config["log_level"],
        rotation="10 MB",
        retention=5,
        compression="zip",
        format="<level>{time:YYYY-MM-DDTHH:mm:ss} {extra[env]} {level} [f:{function}] [L:{line}] {message}</level>",
    )

    logger.configure(extra={"env": "dev"})

    return logger
