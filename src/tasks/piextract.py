#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, List, Tuple
from glob import glob
import datasets
import os


def read_conll_file(file_path: str) -> Dict[str, List[List[str]]]:
    # read all lines in CONLL file and strip trailing newlines
    with open(file_path, "r") as input_file_stream:
        conll_lines = [line.rstrip() for line in input_file_stream]

    # create global dictionary for storing data
    data: Dict[str, List[List[str]]] = {"tokens": [], "ner_tags": []}

    # loop through lines in CONLL file
    for line in conll_lines:
        if line == "-DOCSTART- -X- O O":
            # skip line if DOCSTART encountered
            continue
        elif line == "":
            # append a new list as an empty string denotes
            # the completion of a single annotation
            data["tokens"].append([])
            data["ner_tags"].append([])
        else:
            # in all other cases, split the line and append
            # one token and one NER tag to the final list
            token, ner_tag = line.split(" _ _ ")
            data["tokens"][-1].append(token)
            data["ner_tags"][-1].append(ner_tag)

    return data


def merge_ner_tags(ner_tags: List[List[List[str]]]) -> List[List[Tuple[str, ...]]]:
    # perform a nested zip operation to combine token-level NER tags
    return [list(zip(*ner_tag)) for ner_tag in list(zip(*ner_tags))]


def load_piextract(directory: str) -> datasets.DatasetDict:
    # define task loading order (necessary for multi-label task)
    task_order = ["CollectUse_true", "CollectUse_false", "Share_true", "Share_false"]

    # define global data dictionary
    data: Dict[str, List[Dict[str, List[List[str]]]]] = {"train": [], "test": []}

    # define empty DatasetDict
    combined = datasets.DatasetDict()

    # loop over tasks and CONLL files associated per task
    for task in task_order:
        for conll_file in glob(os.path.join(directory, task, "*.conll03")):
            if os.path.basename(conll_file).startswith("train"):
                split = "train"
            else:
                split = "test"

            # append parsed CONLL file to dictionary by split
            data[split].append(read_conll_file(conll_file))

    # loop over each data split
    for split, data_split in data.items():
        # flatten tokens from all four tasks in this split
        all_tokens = [data_split_subset["tokens"] for data_split_subset in data_split]

        # flatten NER tags from all four tasks in this split
        all_ner_tags = [
            data_split_subset["ner_tags"] for data_split_subset in data_split
        ]

        # ensure that all tokens are exactly the same (assumption for merging)
        assert all([tokens == all_tokens[0] for tokens in all_tokens])

        # merge all NER tags
        merged_ner_tags = merge_ner_tags(all_ner_tags)

        # convert dictionary into HF dataset and insert into DatasetDict
        combined[split] = datasets.Dataset.from_dict(
            {"tokens": all_tokens[0], "ner_tags": merged_ner_tags}
        )

    return combined
