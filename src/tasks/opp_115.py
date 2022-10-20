#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import datasets
import pandas as pd


def load_opp_115(directory: str) -> datasets.DatasetDict:
    # define an empty DatasetDict
    combined = datasets.DatasetDict()

    # define available splits
    splits = ["train", "validation", "test"]

    # define label information
    label_info = datasets.Sequence(
        datasets.ClassLabel(
            names=[
                "Data Retention",
                "Data Security",
                "Do Not Track",
                "First Party Collection/Use",
                "International and Specific Audiences",
                "Introductory/Generic",
                "Policy Change",
                "Practice not covered",
                "Privacy contact information",
                "Third Party Sharing/Collection",
                "User Access, Edit and Deletion",
                "User Choice/Control",
            ]
        )
    )

    # loop over all splits
    for split in splits:
        # read CSV file corresponding to split
        temp_df = pd.read_csv(
            os.path.join(directory, f"{split}_dataset.csv"),
            header=None,
            names=["text", "label"],
        )

        # aggregate all labels per sentence into a unique list
        temp_df = (
            temp_df.groupby("text")
            .agg(dict(label=lambda x: list(set(x))))
            .reset_index()
        )

        # convert temporary dataframe into HF dataset
        dataset = datasets.Dataset.from_pandas(temp_df, preserve_index=False)

        # convert string labels to integers and store feature information
        dataset = dataset.map(
            lambda examples: {
                "label": [
                    label_info.feature.str2int(labels) for labels in examples["label"]
                ]
            },
            batched=True,
        )
        dataset.features["label"] = label_info

        # insert dataset into combined DatasetDict
        combined[split] = dataset

    return combined
