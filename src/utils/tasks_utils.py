#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from datasets import Dataset


def expand_dataset_per_task(ds, tasks):
    # only one label per example, split the data into multiple tasks
    multi_trainset = {"tokens": [], "tags": [], "subtask": []}
    for example in ds:
        for i, st in enumerate(tasks):
            multi_trainset["tokens"].append(example["tokens"])
            multi_trainset["tags"].append([tag[i] for tag in example["tags"]])
            multi_trainset["subtask"].append(st)

    multi_trainset = Dataset.from_dict(multi_trainset)
    return multi_trainset
