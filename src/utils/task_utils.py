#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Dict

from datasets import Dataset, Value, interleave_datasets


def policy_ie_file_mapping(directory: str, filename: str) -> Dict[str, str]:
    # define patterns for file loading
    files = {}
    files["train"] = os.path.join(directory, "train", filename)
    files["validation"] = os.path.join(directory, "valid", filename)
    files["test"] = os.path.join(directory, "test", filename)
    return files


def expand_dataset_per_task(ds, tasks):
    # only one label per example, split the data into multiple tasks
    multi_datasets = {}
    for i, st in enumerate(tasks):
        per_task_dataset = {"tokens": [], "tags": [], "subtask": []}
        for example in ds:
            per_task_dataset["tokens"].append(example["tokens"])
            per_task_dataset["tags"].append([tag[i] for tag in example["tags"]])
            per_task_dataset["subtask"].append(st)
        multi_datasets[st] = Dataset.from_dict(per_task_dataset)
    return multi_datasets


def sorted_interleave_task_datasets(ds_dict, delete_features=False) -> Dataset:
    # make sure the datasets have the same lengths for all tasks
    assert all(
        [
            len(list(ds_dict.values())[0]) == len_val
            for len_val in map(len, ds_dict.values())
        ]
    )
    if delete_features:
        for subtask in ds_dict:
            ds_dict[subtask].features["tags"] = Value("null")
    new_ds = interleave_datasets([dataset for _, dataset in sorted(ds_dict.items())])
    return new_ds
