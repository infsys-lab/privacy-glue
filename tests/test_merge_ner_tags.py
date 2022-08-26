#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tasks.piextract import merge_ner_tags
import pytest


@pytest.mark.parametrize(
    "raw_ner_tags, merged_ner_tags",
    [([[["O", "O", "B-Thing", "I-Thing", "O"], ["O", "O"]
        ], [["O", "B-Thang", "I-Thang", "O", "O"], ["B-Thing", "I-Thing"]]], [[
            ("O", "O"), ("O", "B-Thang"), ("B-Thing", "I-Thang"),
            ("I-Thing", "O"), ("O", "O")
        ], [("O", "B-Thing"), ("O", "I-Thing")]])])
def test_merge_ner_tags(raw_ner_tags, merged_ner_tags):
    assert merge_ner_tags(raw_ner_tags) == merged_ner_tags
