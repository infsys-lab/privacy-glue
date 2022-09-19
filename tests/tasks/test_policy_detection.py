#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tasks.policy_detection import load_policy_detection
import datasets
import os


def test_load_policy_detection():
    # load sample data
    data = load_policy_detection(
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "policy_detection"
        )
    )

    # check that all three splits are included
    assert sorted(data.keys()) == sorted(["train", "validation", "test"])

    # merge train and validation to train to compare against files
    data = datasets.concatenate_datasets(
        [data["train"], data["validation"], data["test"]]
    )

    # define what is expected from the load function
    expected = sorted(
        [
            ("testing once", "Policy"),
            ("testing twice", "Not Policy"),
            ("testing thrice", "Policy"),
        ]
    )

    # assert that we got what is expected
    assert sorted(zip(data["text"], data["label"])) == expected
