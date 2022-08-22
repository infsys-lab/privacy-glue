#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datasets
import pandas as pd


def load_opp_115(directory: str) -> datasets.DatasetDict:
    columns = ["text", "label"]
    df = pd.DataFrame()
    for split in ["train", "validation", "test"]:
        temp_df = pd.read_csv(os.path.join(directory,
                                           "%s_dataset.csv" % split),
                              header=None,
                              names=columns)
        temp_df = temp_df.groupby("text").agg(
            dict(label=lambda x: list(set(x)))).reset_index()
        temp_df["split"] = split
        df = pd.concat([df, temp_df], ignore_index=True)

    dataset = datasets.Dataset.from_pandas(df, preserve_index=False)
    combined = datasets.DatasetDict()
    for split in ["train", "validation", "test"]:
        combined[split] = dataset.filter(
            lambda row: row["split"] == split).remove_columns("split")
    return combined
