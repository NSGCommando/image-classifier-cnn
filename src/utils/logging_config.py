import logging

def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "[%(levelname)s] "
            "[%(threadName)s] "
            "%(asctime)s "
            "%(message)s"
        ),
    )