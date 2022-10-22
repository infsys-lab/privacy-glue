#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tasks.opp_115 import load_opp_115
import pandas as pd
import pytest
import os


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
                (f"{split} check for OPP-115 1", [0]),
                (f"{split} check for OPP-115 2", [1]),
                (f"{split} check for OPP-115 3", [2]),
                (f"{split} check for OPP-115 4", [3]),
                (f"{split} check for OPP-115 5", [4]),
                (f"{split} check for OPP-115 6", [5]),
                (f"{split} check for OPP-115 7", [6]),
                (f"{split} check for OPP-115 8", [7]),
                (f"{split} check for OPP-115 9", [8]),
                (f"{split} check for OPP-115 10", [9]),
                (f"{split} check for OPP-115 11", [10]),
                (f"{split} check for OPP-115 12", [11]),
                (f"{split} check for OPP-115 13", [0, 1]),
                (f"{split} check for OPP-115 14", [4, 5]),
            ]
        )

        # assert that we got what is expected
        assert (
            sorted(zip(data_split["text"], map(sorted, data_split["label"])))
            == expected
        )


def test_load_opp_115_failure(mocker):
    # mock relevant function
    mocker.patch(
        "tasks.opp_115.pd.read_csv",
        return_value=pd.DataFrame(
            {
                "text": ["sample_text_1", "sample_text_1", "sample_text_2"],
                "label": ["sample_label_1", "sample_label_2", "sample_label_3"],
            }
        ),
    )

    # load sample data
    with pytest.raises(ValueError):
        load_opp_115(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "opp_115")
        )
