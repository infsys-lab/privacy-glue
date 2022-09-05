#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tasks.policy_detection import load_policy_detection
import os


def test_load_policy_detection():
    # load sample data
    data = load_policy_detection(
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "policy_detection"
        )
    )

    # check that all three splits are included
    assert set(data.keys()) == {"train", "test"}

    # iterate over splits
    for (split, data_split) in data.items():
        # check that all column names are as expected
        assert data_split.column_names == ["text", "label"]

        # ensure all text is composed of strings and not array-like objects
        assert all([isinstance(text, str) for text in data_split["text"]])

        # ensure all labels are composed of strings and not array-like objects
        assert all([isinstance(label, str) for label in data_split["label"]])
