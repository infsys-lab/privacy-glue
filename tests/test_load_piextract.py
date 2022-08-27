#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tasks.piextract import load_piextract
import os


def test_load_piextract():
    # load sample data
    data = load_piextract(os.path.join(os.path.dirname(__file__), "data", "piextract"))

    # check that all three splits are included
    assert set(data.keys()) == {"train", "test"}

    # iterate over splits
    for (split, data_split) in data.items():
        # check that all column names are as expected
        assert data_split.column_names == ["tokens", "ner_tags"]

        # check that all tokens are present in list objects
        assert all(
            [isinstance(tokens_sample, list) for tokens_sample in data_split["tokens"]]
        )

        # check that all tokens are present as strings
        assert all(
            [
                isinstance(token, str)
                for tokens_sample in data_split["tokens"]
                for token in tokens_sample
            ]
        )

        # check that all NER tags are present in list objects
        assert all(
            [
                isinstance(ner_tags_sample, list)
                for ner_tags_sample in data_split["ner_tags"]
            ]
        )

        # check that all NER tags have two elements
        assert all(
            [
                len(ner_tag) == 4
                and all([isinstance(inner_tag, str) for inner_tag in ner_tag])
                for ner_tag_sample in data_split["ner_tags"]
                for ner_tag in ner_tag_sample
            ]
        )

        # check that tokens and NER tags have the same length
        assert all(
            [
                len(tokens_sample) == len(ner_tag_sample)
                for tokens_sample, ner_tag_sample in zip(
                    data_split["tokens"], data_split["ner_tags"]
                )
            ]
        )
