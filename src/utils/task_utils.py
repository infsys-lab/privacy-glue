#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict
import os


def policy_ie_file_mapping(directory: str, filename: str) -> Dict[str, str]:
    # define patterns for file loading
    files = {}
    files["train"] = os.path.join(directory, "train", filename)
    files["validation"] = os.path.join(directory, "valid", filename)
    files["test"] = os.path.join(directory, "test", filename)
    return files
