#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tasks.opp_115 import load_opp_115
import os


def test_load_opp_115():
    # load sample data
    data = load_opp_115(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "opp_115")
    )

    # check that all three splits are included
    assert set(data.keys()) == {"train", "validation", "test"}

    # iterate over splits
    for (split, data_split) in data.items():
        # check that all column names are as expected
        assert data_split.column_names == ["text", "label"]

        # define what is expected from the load function
        expected = set(
            [
                (
                    f"{split} check for OPP-115",
                    frozenset([f"label-{split}-1", f"label-{split}-2"]),
                ),
                (
                    f"another {split} check for OPP-115",
                    frozenset([f"label-{split}-2"]),
                ),
            ]
        )

        # assert that we got what is expected
        assert (
            set(zip(data_split["text"], map(frozenset, data_split["label"])))
            == expected
        )
