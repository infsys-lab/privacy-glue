#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

# create formatter
FORMATTER = logging.Formatter(
    fmt="%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S"
)


def init_logger(logger: logging.Logger, level: str) -> None:
    # set logger level
    logger.setLevel(level)

    # set output stream to stderr
    stderr_handler = logging.StreamHandler()
    stderr_handler.setLevel(level)
    stderr_handler.setFormatter(FORMATTER)

    # add stream to logger
    logger.addHandler(stderr_handler)


def add_file_handler(logger: logging.Logger, level: str, filename: str) -> None:
    # create file handler
    file_handler = logging.FileHandler(filename)

    # set logging level
    file_handler.setLevel(level)

    # set formatter
    file_handler.setFormatter(FORMATTER)

    # add file handler to logger
    logger.addHandler(file_handler)


def remove_all_file_handlers(logger: logging.Logger) -> None:
    # loop over all handlers in a copy of logger's handlers
    for handler in logger.handlers[:]:
        # remove handler if it is a file-handler
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
