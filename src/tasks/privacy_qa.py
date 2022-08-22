#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import datasets
import os


def load_privacy_qa(directory: str) -> datasets.DatasetDict:
    # load and process the train dataset
    train_df = pd.read_csv(os.path.join(directory, "policy_train.tsv"),
                           sep="\t")
    train_df = train_df[["DocID", "QueryID", "Query", "Segment", "Label"]]
    train_df["Label"] = train_df["Label"].apply(
        lambda x: x == "Relevant").astype(int)
    train_df = train_df.groupby(["DocID", "QueryID"]).agg({
        "Query":
        lambda x: x.unique(),
        "Segment":
        lambda x: x.tolist(),
        "Label":
        lambda x: x.tolist()
    })
    train_df = train_df.reset_index()[["QueryID", "Query", "Segment", "Label"]]
    train_df = train_df.rename(columns={
        "Query": "question",
        "Segment": "text",
        "Label": "label"
    })

    # load and process the train meta dataset
    train_meta_df = pd.read_csv(os.path.join(directory,
                                             "train_opp_annotations.tsv"),
                                sep="\t")
    train_meta_df = train_meta_df.drop(["Folder", "DocID", "Split"], axis=1)
    train_meta_df["opp_category"] = train_meta_df[[
        "first", "third", "datasecurity", "dataretention", "user_access",
        "user_choice", "other"
    ]].apply(lambda row: row[row == 1].index.tolist(), axis=1)
    assert set(train_df["QueryID"].tolist()) == set(
        train_meta_df["QueryID"].tolist())
    train_df = train_df.merge(train_meta_df[["QueryID", "opp_category"]],
                              on="QueryID")
    train_dataset = datasets.Dataset.from_pandas(train_df.drop("QueryID",
                                                               axis=1),
                                                 preserve_index=False)
    for sample in train_dataset:
        assert len(sample["text"]) == len(sample["label"])

    # work on the test dataset
    test_df = pd.read_csv(os.path.join(directory, "policy_test.tsv"), sep="\t")
    test_df = test_df[["DocID", "QueryID", "Query", "Segment", "Any_Relevant"]]
    test_df = test_df.rename(columns={"Any_Relevant": "Label"})
    test_df["Label"] = test_df["Label"].apply(
        lambda x: x == "Relevant").astype(int)
    test_df = test_df.groupby(["DocID", "QueryID"]).agg({
        "Query":
        lambda x: x.unique(),
        "Segment":
        lambda x: x.tolist(),
        "Label":
        lambda x: x.tolist()
    })
    test_df = test_df.reset_index()[["Query", "QueryID", "Segment", "Label"]]
    test_df = test_df.rename(columns={
        "Query": "question",
        "Segment": "text",
        "Label": "label"
    })

    # load and process the test meta dataset
    test_meta_df = pd.read_csv(os.path.join(directory,
                                            "test_opp_annotations.tsv"),
                               sep="\t")
    test_meta_df = test_meta_df.drop(["Folder", "DocID", "Split"], axis=1)
    test_meta_df["opp_category"] = test_meta_df[[
        "first", "third", "datasecurity", "dataretention", "user_access",
        "user_choice", "other"
    ]].apply(lambda row: row[row == 1].index.tolist(), axis=1)
    assert set(test_df["QueryID"].tolist()) == set(
        test_meta_df["QueryID"].tolist())
    test_df = test_df.merge(test_meta_df[["QueryID", "opp_category"]],
                            on="QueryID")
    test_dataset = datasets.Dataset.from_pandas(test_df.drop("QueryID",
                                                             axis=1),
                                                preserve_index=False)
    for sample in test_dataset:
        assert len(sample["text"]) == len(sample["label"])

    # return a combination of train and test datasets
    return datasets.DatasetDict({"train": train_dataset, "test": test_dataset})
