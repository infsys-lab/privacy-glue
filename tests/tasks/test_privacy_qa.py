#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tasks.privacy_qa import load_privacy_qa
import datasets
import os


def test_load_privacy_qa():
    # load sample data
    data = load_privacy_qa(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "privacy_qa")
    )

    # check that all three splits are included
    assert set(data.keys()) == {"train", "validation", "test"}

    # merge train and validation to train to compare against files
    data["train"] = datasets.concatenate_datasets([data["train"], data["validation"]])
    del data["validation"]

    # iterate over splits
    for (split, data_split) in data.items():
        # check that all column names are as expected
        assert data_split.column_names == ["question", "text", "label"]

        # define what is expected from the load function
        expected = set(
            [
                (
                    f"{split} question for PrivacyQA?",
                    f"{split} answer for PrivacyQA",
                    f"label-{split}-1",
                ),
                (
                    f"another {split} question for PrivacyQA?",
                    f"another {split} answer for PrivacyQA",
                    f"label-{split}-2",
                ),
            ]
        )

        # assert that we got what is expected
        assert (
            set(zip(data_split["question"], data_split["text"], data_split["label"]))
            == expected
        )
