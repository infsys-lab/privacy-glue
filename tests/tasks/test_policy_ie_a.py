#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tasks.policy_ie_a import load_policy_ie_a
import datasets
import pytest
import os


def test_load_policy_ie_a():
    # load sample data
    data = load_policy_ie_a(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "policy_ie_a")
    )

    # check that all three splits are included
    assert sorted(data.keys()) == sorted(["train", "validation", "test"])

    # iterate over splits
    for (split, data_split) in data.items():
        # check that all column names are as expected
        assert data_split.column_names == ["text", "label"]

        # define what is expected from the load function
        expected = sorted(
            [
                (f"{split} check for PolicyIE-A 1", 0),
                (f"{split} check for PolicyIE-A 2", 1),
                (f"{split} check for PolicyIE-A 3", 2),
                (f"{split} check for PolicyIE-A 4", 3),
                (f"{split} check for PolicyIE-A 5", 4),
            ]
        )

        # assert that we got what is expected
        assert sorted(zip(data_split["text"], data_split["label"])) == expected


def test_load_policy_ie_a_failure(mocker):
    # create mocked dataset dictionary
    combined = datasets.DatasetDict()
    for split in ["train", "validation", "test"]:
        combined["train"] = datasets.Dataset.from_dict(
            {"text": ["some_text_or_label_1", "some_text_or_label_2"]}
        )

    # mock relevant function
    mocker.patch("tasks.policy_ie_a.datasets.load_dataset", return_value=combined)

    # load sample data
    with pytest.raises(ValueError):
        load_policy_ie_a(
            os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "data", "policy_ie_a"
            )
        )
