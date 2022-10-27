#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import tempfile

import pytest

from utils.logging_utils import (
    FORMATTER,
    add_file_handler,
    init_logger,
    remove_all_file_handlers,
)


@pytest.mark.parametrize(
    "level",
    ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
)
def test_init_logger(level):
    logger = logging.getLogger(level)
    init_logger(logger, level)
    assert logger.level == getattr(logging, level)
    assert isinstance(logger.handlers[-1], logging.StreamHandler)
    assert logger.handlers[-1].level == getattr(logging, level)
    assert logger.handlers[-1].formatter == FORMATTER


@pytest.mark.parametrize(
    "level, filename",
    [
        ("DEBUG", "debug.txt"),
        ("INFO", "info.txt"),
        ("WARNING", "warning.txt"),
        ("ERROR", "error.txt"),
        ("CRITICAL", "critical.txt"),
    ],
)
def test_add_file_handler(level, filename):
    logger = logging.getLogger(level)
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, filename)
        add_file_handler(logger, level, path)
        assert isinstance(logger.handlers[-1], logging.FileHandler)
        assert logger.handlers[-1].level == getattr(logging, level)
        assert logger.handlers[-1].formatter == FORMATTER
        assert logger.handlers[-1].baseFilename == path


@pytest.mark.parametrize(
    "level, filename",
    [
        ("DEBUG", "debug.txt"),
        ("INFO", "info.txt"),
        ("WARNING", "warning.txt"),
        ("ERROR", "error.txt"),
        ("CRITICAL", "critical.txt"),
    ],
)
def test_remove_all_file_handlers(level, filename):
    logger = logging.getLogger(level)
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, filename)
        add_file_handler(logger, level, path)
        remove_all_file_handlers(logger)
        assert not any(
            [isinstance(handler, logging.FileHandler) for handler in logger.handlers]
        )
