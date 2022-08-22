#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datasets
import os


def load_policy_qa(directory: str) -> datasets.DatasetDict:
    data_files = {}
    data_files["train"] = os.path.join(directory, "train.jsonl")
    data_files["validation"] = os.path.join(directory, "dev.jsonl")
    data_files["test"] = os.path.join(directory, "test.jsonl")
    return datasets.load_dataset("json", data_files=data_files).map(
        lambda example:
        {"question_type": example["question_type"].split("|||")})
