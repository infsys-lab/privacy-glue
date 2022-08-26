#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tasks.policy_ie_b import load_policy_ie_b
import os


def test_load_policy_ie_b():
    # load sample data
    data = load_policy_ie_b(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data",
                     "policy_ie_b"))

    # check that all three splits are included
    assert set(data.keys()) == {"train", "validation", "test"}

    # iterate over splits
    for (split, data_split) in data.items():
        # check that all column names are as expected
        assert data_split.column_names == ["tokens", "ner_tags"]

        # check that tokens and NER tags have the same length
        assert len(data_split["tokens"]) == len(data_split["ner_tags"])

        # check that all NER tags have two elements
        for ner_tag_sample in data_split["ner_tags"]:
            for ner_tag in ner_tag_sample:
                assert len(ner_tag) == 2
