#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Dict, cast

import datasets


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
    tokens = datasets.load_dataset("text", data_files=file_mapping(directory, "seq.in"))

    # since this is task A, only load labels
    labels = datasets.load_dataset(
        "text", data_files=file_mapping(directory, "label")
    ).rename_column("text", "label")

    # retrieve unique labels and add them to datastructure
    unique_labels = list(set(labels["train"]["label"]))
    label_info = datasets.ClassLabel(
        num_classes=len(unique_labels), names=unique_labels
    )

    # mypy-related specification to sub-type
    tokens = cast(datasets.DatasetDict, tokens)
    labels = cast(datasets.DatasetDict, labels)

    # zip together data
    for split in ["train", "validation", "test"]:
        ds = datasets.concatenate_datasets([tokens[split], labels[split]], axis=1)
        ds.features["label"] = label_info
        combined[split] = ds

    return combined
