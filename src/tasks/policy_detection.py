#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import datasets
import json
import os


def load_policy_detection(directory: str) -> datasets.DatasetDict:
    # initialize DatasetDict object
    combined = datasets.DatasetDict()

    # read csv file and choose subset of columns
    df = pd.read_csv(os.path.join(directory, "1301_dataset.csv"), index_col=0)
    df = df[["policy_text", "is_policy"]]

    # read json file with split ids
    with open(os.path.join(directory, "split_ids.json"), "r") as input_file_stream:
        split_ids = json.load(input_file_stream)

    # assert that all lengths make sense
    assert sum([len(ids) for ids in split_ids.values()]) == df.shape[0]

    # replace labels from boolean to strings for consistency
    df["is_policy"] = df["is_policy"].replace({True: "Policy", False: "Not Policy"})

    # rename columns for consistency
    df = df.rename(columns={"policy_text": "text", "is_policy": "label"})

    # convert into HF datasets
    dataset = datasets.Dataset.from_pandas(df, preserve_index=True)

    # split into train and test sets
    for split, ids in split_ids.items():
        combined[split] = dataset.filter(
            lambda example: example["__index_level_0__"] in ids
        ).remove_columns("__index_level_0__")

    return combined
