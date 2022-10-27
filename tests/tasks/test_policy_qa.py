#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from tasks.policy_qa import load_policy_qa


def test_load_policy_qa():
    # load sample data
    data = load_policy_qa(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "policy_qa")
    )

    # check that all three splits are included
    assert sorted(data.keys()) == sorted(["train", "test", "validation"])

    # iterate over splits
    for (split, data_split) in data.items():
        # check that all column names are as expected
        assert data_split.column_names == [
            "id",
            "title",
            "context",
            "question",
            "answers",
        ]

        # define what is expected from the load function
        expected = sorted(
            [
                (
                    "some_id",
                    f"{split}.com",
                    f"{split} answer for PolicyQA",
                    f"{split} question for PolicyQA?",
                    (0,),
                    (f"{split} answer for PolicyQA",),
                ),
                (
                    "another_id",
                    f"{split}.com",
                    f"another {split} answer for PolicyQA",
                    f"another {split} question for PolicyQA?",
                    (0,),
                    (f"another {split} answer for PolicyQA",),
                ),
            ]
        )

        # assert that we got what is expected
        assert (
            sorted(
                zip(
                    data_split["id"],
                    data_split["title"],
                    data_split["context"],
                    data_split["question"],
                    tuple(
                        [
                            tuple(answer["answer_start"])
                            for answer in data_split["answers"]
                        ]
                    ),
                    tuple([tuple(answer["text"]) for answer in data_split["answers"]]),
                )
            )
            == expected
        )
