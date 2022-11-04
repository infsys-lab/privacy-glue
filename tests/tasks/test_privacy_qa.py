#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import datasets
import pandas as pd
import pytest

from tasks.privacy_qa import load_privacy_qa


def test_load_privacy_qa():
    # load sample data
    data = load_privacy_qa(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "privacy_qa")
    )

    # check that all three splits are included
    assert sorted(data.keys()) == sorted(["train", "validation", "test"])

    # merge train and validation to train to compare against files
    data["train"] = datasets.concatenate_datasets([data["train"], data["validation"]])
    del data["validation"]

    # iterate over splits
    for (split, data_split) in data.items():
        # check that all column names are as expected
        assert data_split.column_names == ["question", "text", "label"]

        # define what is expected from the load function
        expected = sorted(
            [
                (
                    f"{split} question for PrivacyQA?",
                    f"{split} answer for PrivacyQA",
                    1,
                ),
                (
                    f"another {split} question for PrivacyQA?",
                    f"another {split} answer for PrivacyQA",
                    0,
                ),
            ]
        )

        # assert that we got what is expected
        assert (
            sorted(zip(data_split["question"], data_split["text"], data_split["label"]))
            == expected
        )


def test_load_privacy_qa_failure(mocker):
    # mock relevant function
    mocker.patch(
        "tasks.privacy_qa.pd.read_csv",
        return_value=pd.DataFrame(
            {
                "Query": ["some_query_1", "some_query_2"],
                "Segment": ["some_segment_1", "some_segment_2"],
                "Label": ["some_label_1", "some_label_2"],
                "Any_Relevant": ["some_label_1", "some_label_2"],
            }
        ),
    )

    # load sample data
    with pytest.raises(ValueError):
        load_privacy_qa(
            os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "data", "privacy_qa"
            )
        )
