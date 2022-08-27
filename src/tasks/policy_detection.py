#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import datasets
import os


def load_policy_detection(directory: str) -> datasets.DatasetDict:
    # read csv file and choose subset of columns
    df = pd.read_csv(os.path.join(directory, "1301_dataset.csv"), index_col=0)
    df = df[["policy_text", "is_policy"]]

    # replace labels from boolean to strings for consistency
    df["is_policy"] = df["is_policy"].replace({True: "Policy", False: "Not Policy"})

    # rename columns for consistency
    df = df.rename(columns={"policy_text": "text", "is_policy": "label"})

    # convert into HF datasets
    dataset = datasets.Dataset.from_pandas(df, preserve_index=False)

    # split into train and test sets
    dataset = dataset.train_test_split(test_size=0.3, seed=42)

    return dataset
