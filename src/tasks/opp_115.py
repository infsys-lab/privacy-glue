#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datasets
import pandas as pd
import os


def load_opp_115(directory: str) -> datasets.DatasetDict:
    # define an empty DatasetDict
    combined = datasets.DatasetDict()

    # define available splits
    splits = ["train", "validation", "test"]
    # loop over all splits
    for split in splits:
        # read CSV file corresponding to split
        temp_df = pd.read_csv(
            os.path.join(directory, "%s_dataset.csv" % split),
            header=None,
            names=["text", "label"],
        )
        unique_labels = temp_df.label.unique()
        # aggregate all labels per sentence into a unique list
        temp_df = (
            temp_df.groupby("text")
            .agg(
                dict(
                    label=lambda x: [float(label in set(x)) for label in unique_labels]
                )
            )
            .reset_index()
        )
        # convert temporary dataframe into HF dataset
        dataset = datasets.Dataset.from_pandas(temp_df, preserve_index=False)
        label_info = datasets.ClassLabel(
            num_classes=len(unique_labels), names=unique_labels
        )
        dataset.features["label"] = label_info
        # insert dataset into combined DatasetDict
        combined[split] = dataset
    return combined
