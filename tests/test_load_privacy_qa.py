#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tasks.privacy_qa import load_privacy_qa
import os


def test_load_privacy_qa():
    # load sample data
    data = load_privacy_qa(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data",
                     "privacy_qa"))

    # check that all three splits are included
    assert set(data.keys()) == {"train", "test"}

    # iterate over splits
    for (split, data_split) in data.items():
        # check that all column names are as expected
        assert data_split.column_names == ["question", "text", "label"]

        # ensure all questions are composed of strings and not array-like
        # objects
        assert all([isinstance(text, str) for text in data_split["question"]])

        # ensure all text is composed of strings and not array-like objects
        assert all([isinstance(text, str) for text in data_split["text"]])

        # ensure all labels are composed of strings and not array-like objects
        assert all([isinstance(text, str) for text in data_split["label"]])
