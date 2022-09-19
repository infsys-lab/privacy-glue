#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tasks.policy_ie_a import load_policy_ie_a
import os


def test_load_policy_ie_a():
    # load sample data
    data = load_policy_ie_a(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "policy_ie_a")
    )

    # check that all three splits are included
    assert sorted(data.keys()) == sorted(["train", "validation", "test"])

    # iterate over splits
    for (split, data_split) in data.items():
        # check that all column names are as expected
        assert data_split.column_names == ["text", "label"]

        # define what is expected from the load function
        expected = sorted(
            [
                (
                    f"{split} check for PolicyIE-A",
                    f"label-{split}-1",
                ),
                (
                    f"another {split} check for PolicyIE-A",
                    f"label-{split}-2",
                ),
            ]
        )

        # assert that we got what is expected
        assert sorted(zip(data_split["text"], data_split["label"])) == expected
