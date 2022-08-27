#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import datasets
import os


def load_privacy_qa(directory: str) -> datasets.DatasetDict:
    # load and process the train dataset
    train_df = pd.read_csv(os.path.join(directory, "policy_train.tsv"), sep="\t")
    train_df = train_df[["Query", "Segment", "Label"]].rename(
        columns={"Query": "question", "Segment": "text", "Label": "label"}
    )
    train_dataset = datasets.Dataset.from_pandas(train_df, preserve_index=False)

    # work on the test dataset
    test_df = pd.read_csv(os.path.join(directory, "policy_test.tsv"), sep="\t")
    test_df = test_df[["Query", "Segment", "Any_Relevant"]].rename(
        columns={"Query": "question", "Segment": "text", "Any_Relevant": "label"}
    )
    test_dataset = datasets.Dataset.from_pandas(test_df, preserve_index=False)

    # concatenate both datasets
    combined = datasets.DatasetDict({"train": train_dataset, "test": test_dataset})

    return combined
