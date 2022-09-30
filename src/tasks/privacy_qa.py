#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import datasets
import pandas as pd


def load_privacy_qa(directory: str) -> datasets.DatasetDict:
    # load and process the train dataset
    train_df = pd.read_csv(os.path.join(directory, "policy_train.tsv"), sep="\t")
    train_df = train_df[["Query", "Segment", "Label"]].rename(
        columns={"Query": "question", "Segment": "text", "Label": "label"}
    )
    # collect information about label
    unique_labels = list(set(train_df["label"]))
    label_info = datasets.ClassLabel(
        num_classes=len(unique_labels), names=unique_labels
    )
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

    # distribute info about labels
    for split in ["train", "validation", "test"]:
        combined[split].features["label"] = label_info

    return combined
