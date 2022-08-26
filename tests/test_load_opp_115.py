#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tasks.opp_115 import load_opp_115
import os


def test_load_opp_115():
    # load sample data
    data = load_opp_115(
        os.path.join(os.path.dirname(__file__), "data", "opp_115"))

    # check that all three splits are included
    assert set(data.keys()) == {"train", "validation", "test"}

    # iterate over splits
    for (split, data_split) in data.items():
        # check that all column names are as expected
        assert data_split.column_names == ["text", "label"]

        # ensure all text is composed of strings and not array-like objects
        assert all([isinstance(text, str) for text in data_split["text"]])

        # ensure all labels are composed of strings and not array-like objects
        assert all([isinstance(label, list) for label in data_split["label"]])
