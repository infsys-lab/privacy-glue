#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tasks.policy_ie_b import load_policy_ie_b
import os


def test_load_policy_ie_b():
    # load sample data
    data = load_policy_ie_b(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "policy_ie_b")
    )

    # check that all three splits are included
    assert set(data.keys()) == {"train", "validation", "test"}

    # iterate over splits
    for (split, data_split) in data.items():
        # check that all column names are as expected
        assert data_split.column_names == ["tokens", "ner_tags"]

        # define what is expected from the load function
        expected = set(
            [
                (
                    (f"{split}", "check", "for", "PolicyIE-B"),
                    (
                        ("O", "O"),
                        ("O", "O"),
                        ("O", "O"),
                        (f"label-{split}-1", f"label-{split}-3"),
                    ),
                ),
                (
                    ("another", f"{split}", "check", "for", "PolicyIE-B"),
                    (
                        ("O", "O"),
                        ("O", "O"),
                        ("O", "O"),
                        ("O", "O"),
                        (f"label-{split}-2", f"label-{split}-4"),
                    ),
                ),
            ]
        )

        # assert that we got what is expected
        assert (
            set(
                zip(
                    map(tuple, data_split["tokens"]),
                    [
                        tuple(map(tuple, ner_tags))
                        for ner_tags in data_split["ner_tags"]
                    ],
                )
            )
            == expected
        )
