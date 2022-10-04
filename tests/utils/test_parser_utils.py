#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils.parser_utils import dir_path, file_path
import tempfile
import argparse
import pytest


def test_dir_path():
    # check when directory exists
    with tempfile.TemporaryDirectory() as tmp_dir:
        assert dir_path(tmp_dir) == tmp_dir

    # check when directory does not exist
    with pytest.raises(argparse.ArgumentTypeError):
        dir_path(tmp_dir)


def test_file_path():
    # check when file exists
    with tempfile.TemporaryFile() as tmp_file:
        assert file_path(tmp_file.name) == tmp_file.name

    # check when file does not exist
    with pytest.raises(argparse.ArgumentTypeError):
        file_path(tmp_file.name)
