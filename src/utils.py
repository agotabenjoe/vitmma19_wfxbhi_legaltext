import logging
import os

def setup_logger(log_path=None, level=logging.INFO):
    """
    Sets up a logger that logs to both stdout and a file (if log_path is given).
    """
    logger = logging.getLogger()
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    # Remove all handlers
    while logger.handlers:
        logger.handlers.pop()

    # Stream handler (stdout)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # File handler
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
