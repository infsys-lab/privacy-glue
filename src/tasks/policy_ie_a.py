#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict
import datasets
import os


def file_mapping(directory: str, filename: str) -> Dict[str, str]:
    # define patterns for file loading
    files = {}
    files["train"] = os.path.join(directory, "train", filename)
    files["validation"] = os.path.join(directory, "valid", filename)
    files["test"] = os.path.join(directory, "test", filename)

    return files


def load_policy_ie_a(directory: str) -> datasets.DatasetDict:
    # initialize DatasetDict object
    combined = datasets.DatasetDict()

    # load tokens which are common for all sub-tasks
    tokens = datasets.load_dataset(
        "text", data_files=file_mapping(directory, "seq.in")).map(
            lambda example: {"tokens": example["text"].split()},
            remove_columns=["text"])

    # since this is task A, only load labels
    labels = datasets.load_dataset(
        "text",
        data_files=file_mapping(directory,
                                "label")).rename_column("text", "label")

    # zip together data
    for (split,
            tokens_split), (_, labels_split) in zip(tokens.items(),
                                                    labels.items()):
        combined[split] = datasets.concatenate_datasets(
            [tokens_split, labels_split], axis=1)

    return combined
