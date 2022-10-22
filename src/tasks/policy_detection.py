#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import datasets
import pandas as pd

LABELS = ["Not Policy", "Policy"]


def load_policy_detection(directory: str) -> datasets.DatasetDict:
    # initialize DatasetDict object
    combined = datasets.DatasetDict()

    # read csv file and choose subset of columns
    df = pd.read_csv(os.path.join(directory, "1301_dataset.csv"), index_col=0)
    df = df[["policy_text", "is_policy"]]

    # replace labels from boolean to strings for consistency
    df["is_policy"] = df["is_policy"].replace({True: "Policy", False: "Not Policy"})

    # rename columns for consistency
    df = df.rename(columns={"policy_text": "text", "is_policy": "label"})

    # convert into HF datasets
    dataset = datasets.Dataset.from_pandas(df, preserve_index=False)
    label_info = datasets.ClassLabel(names=LABELS)

    # make split using HF datasets internal methods
    train_test_dataset_dict = dataset.train_test_split(test_size=0.3, seed=42)
    train_valid_dataset_dict = train_test_dataset_dict["train"].train_test_split(
        test_size=0.15, seed=42
    )

    # manually assign them to another DatasetDict
    combined["train"] = train_valid_dataset_dict["train"]
    combined["validation"] = train_valid_dataset_dict["test"]
    combined["test"] = train_test_dataset_dict["test"]

    # collect and distribute information about label
    for split in ["train", "validation", "test"]:
        combined[split] = combined[split].map(
            lambda examples: {
                "label": [label_info.str2int(label) for label in examples["label"]]
            },
            batched=True,
        )
        combined[split].features["label"] = label_info

    return combined
