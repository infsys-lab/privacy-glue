#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tasks.policy_qa import load_policy_qa
import os


def test_load_policy_qa():
    # load sample data
    data = load_policy_qa(
        os.path.join(os.path.dirname(__file__), "data", "policy_qa"))

    # check that all three splits are included
    assert set(data.keys()) == {"train", "test", "validation"}

    # iterate over splits
    for (split, data_split) in data.items():
        # check that all column names are as expected
        assert data_split.column_names == [
            "id", "title", "context", "question", "answers"
        ]

        # check that data types are as expected
        assert all([isinstance(idx, str) for idx in data_split["id"]])
        assert all([isinstance(title, str) for title in data_split["title"]])
        assert all(
            [isinstance(context, str) for context in data_split["context"]])
        assert all([
            set(answers.keys()) == {"answer_start", "text"}
            for answers in data_split["answers"]
        ])
        assert all([
            isinstance(answers["answer_start"], list)
            and isinstance(answers["text"], list)
            for answers in data_split["answers"]
        ])
