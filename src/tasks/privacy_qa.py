#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import datasets
import pandas as pd

LABELS = ["Irrelevant", "Relevant"]


def load_privacy_qa(directory: str) -> datasets.DatasetDict:
    # load and process the train dataset
    train_df = pd.read_csv(os.path.join(directory, "policy_train.tsv"), sep="\t")
    train_df = train_df[["Query", "Segment", "Label"]].rename(
        columns={"Query": "question", "Segment": "text", "Label": "label"}
    )

    # collect information about label
    label_info = datasets.ClassLabel(names=LABELS)
    train_dataset = datasets.Dataset.from_pandas(train_df, preserve_index=False)

    # work on the test dataset
    test_df = pd.read_csv(os.path.join(directory, "policy_test.tsv"), sep="\t")
    test_df = test_df[["Query", "Segment", "Any_Relevant"]].rename(
        columns={"Query": "question", "Segment": "text", "Any_Relevant": "label"}
    )
    test_dataset = datasets.Dataset.from_pandas(test_df, preserve_index=False)

    # make split using HF datasets internal methods
    train_valid_dataset_dict = train_dataset.train_test_split(test_size=0.15, seed=42)

    # concatenate both datasets
    combined = datasets.DatasetDict(
        {
            "train": train_valid_dataset_dict["train"],
            "validation": train_valid_dataset_dict["test"],
            "test": test_dataset,
        }
    )

    # map labels to integers and add feature information
    for split in ["train", "validation", "test"]:
        combined[split] = combined[split].map(
            lambda examples: {
                "label": [label_info.str2int(label) for label in examples["label"]]
            },
            batched=True,
        )
        combined[split].features["label"] = label_info

    return combined
