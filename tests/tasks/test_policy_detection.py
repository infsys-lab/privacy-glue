#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tasks.policy_detection import load_policy_detection
import pandas as pd
import pytest
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

    # assert that we got what is expected
    assert data["train"][0] == {"text": "testing twice", "label": 0}
    assert data["validation"][0] == {"text": "testing once", "label": 1}
    assert data["test"][0] == {"text": "testing thrice", "label": 1}


def test_load_policy_detection_failure(mocker):
    # mock relevant function
    mocker.patch(
        "tasks.policy_detection.pd.read_csv",
        return_value=pd.DataFrame(
            {
                "policy_text": ["some_text_1", "some_text_2", "some_text_3"],
                "is_policy": ["random_label_1", "random_label_2", "random_label_3"],
            }
        ),
    )

    # load sample data
    with pytest.raises(ValueError):
        load_policy_detection(
            os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "data", "policy_detection"
            )
        )
