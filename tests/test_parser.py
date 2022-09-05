#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from parser import ModelArguments, DataArguments, TrainingArguments, get_parser


def test_get_parser():
    parser = get_parser()
    assert set(parser.dataclass_types) == {
        ModelArguments,
        DataArguments,
        TrainingArguments,
    }
