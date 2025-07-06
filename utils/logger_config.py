import logging
import os
import sys
from logging.handlers import RotatingFileHandler


class SafeJobFormatter(logging.Formatter):
    def format(self, record):
        if not hasattr(record, 'job_id'):
            record.job_id = '-'
        return super().format(record)


def get_job_logger(base_logger, job_id):
    return logging.LoggerAdapter(base_logger, {"job_id": job_id})


def setup_logger(name=None, log_file=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = SafeJobFormatter(
        '%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d [%(job_id)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
