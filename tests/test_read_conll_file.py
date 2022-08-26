#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from glob import glob
from tasks.piextract import read_conll_file
import os


def test_read_conll_file():
    for conll_file in glob(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "data",
                         "piextract", "**", "*.conll03")):
        # read CONLL file
        data = read_conll_file(conll_file)

        # check their keys make sense
        assert set(data.keys()) == {"tokens", "ner_tags"}

        # check that all tokens are present in list objects
        assert all([
            isinstance(tokens_sample, list)
            for tokens_sample in data["tokens"]
        ])

        # check that all tokens are present as strings
        assert all([
            isinstance(token, str) for tokens_sample in data["tokens"]
            for token in tokens_sample
        ])

        # check that all NER tags are present in list objects
        assert all([
            isinstance(ner_tags_sample, list)
            for ner_tags_sample in data["ner_tags"]
        ])

        # check that all NER tags have two elements
        assert all([
            isinstance(ner_tag, str)
            for ner_tag_sample in data["ner_tags"]
            for ner_tag in ner_tag_sample
        ])

        # check that tokens and NER tags have the same length
        assert all([
            len(tokens_sample) == len(ner_tag_sample) for tokens_sample,
            ner_tag_sample in zip(data["tokens"], data["ner_tags"])
        ])
