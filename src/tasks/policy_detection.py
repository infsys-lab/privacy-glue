#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import datasets
import os


def load_policy_detection(directory: str) -> datasets.DatasetDict:
    df = pd.read_csv(os.path.join(directory, "1301_dataset.csv"), index_col=0)
    df = df[["policy_text", "is_policy"]]
    df["is_policy"] = df["is_policy"].astype(int)
    df = df.rename(columns={"policy_text": "text", "is_policy": "label"})
    dataset = datasets.Dataset.from_pandas(df, preserve_index=False)
    return dataset.train_test_split(test_size=0.3, seed=42)
