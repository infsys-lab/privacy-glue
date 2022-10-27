#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from tasks.opp_115 import load_opp_115


def test_load_opp_115():
    # load sample data
    data = load_opp_115(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "opp_115")
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
                    f"{split} check for OPP-115",
                    sorted([f"label-{split}-1", f"label-{split}-2"]),
                ),
                (
                    f"another {split} check for OPP-115",
                    sorted([f"label-{split}-2"]),
                ),
            ]
        )

        # assert that we got what is expected
        assert (
            sorted(zip(data_split["text"], map(sorted, data_split["label"])))
            == expected
        )
