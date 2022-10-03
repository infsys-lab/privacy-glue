#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from tasks.policy_detection import load_policy_detection


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

    assert data["train"][0] == {"text": "testing twice", "label": "Not Policy"}
    assert data["validation"][0] == {"text": "testing once", "label": "Policy"}
    assert data["test"][0] == {"text": "testing thrice", "label": "Policy"}
