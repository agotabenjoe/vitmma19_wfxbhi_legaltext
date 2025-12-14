import logging
import sys

def setup_logger(level=logging.INFO):
    """
    Sets up a logger that logs to stdout (for Docker compatibility).
    """
    logger = logging.getLogger("legaltext")
    logger.setLevel(level)
    
    # Only add handler if none exist
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        # Stream handler (stdout) - Docker captures stdout
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    return logger

# Create a module-level logger for all files to import
logger = setup_logger()
