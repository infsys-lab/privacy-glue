#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, List
from glob import glob
import datasets
import os


def read_conll_file(path: str) -> Dict[str, List[List[str]]]:
    with open(path, "r") as input_file_stream:
        conll_lines = [line.rstrip() for line in input_file_stream]
    data = {"tokens": [], "ner_tags": []}
    for line in conll_lines:
        if line == "-DOCSTART- -X- O O":
            continue
        elif line == "":
            data["tokens"].append([])
            data["ner_tags"].append([])
        else:
            token, ner_tag = line.split(" _ _ ")
            data["tokens"][-1].append(token)
            data["ner_tags"][-1].append(ner_tag)
    return data


def merge_ner_tags(ner_tags: List[List[str]]) -> List[List[str]]:
    return [list(zip(*ner_tag)) for ner_tag in list(zip(*ner_tags))]


def load_piextract(directory: str) -> datasets.DatasetDict:
    data = {"train": [], "test": []}
    combined = datasets.DatasetDict()
    task_order = [
        "CollectUse_true", "CollectUse_false", "Share_true", "Share_false"
    ]

    for task in task_order:
        for conll_file in glob(os.path.join(directory, task, "*.conll03")):
            if os.path.basename(conll_file).startswith("train"):
                split = "train"
            else:
                split = "test"
            data[split].append(read_conll_file(conll_file))

    for split, data_split in data.items():
        all_tokens = [
            data_split_subset["tokens"] for data_split_subset in data_split
        ]
        all_ner_tags = [
            data_split_subset["ner_tags"] for data_split_subset in data_split
        ]
        assert all([tokens == all_tokens[0] for tokens in all_tokens])
        merged_ner_tags = merge_ner_tags(all_ner_tags)
        combined[split] = datasets.Dataset.from_dict({
            "tokens":
            all_tokens[0],
            "ner_tags":
            merged_ner_tags
        })

    return combined
