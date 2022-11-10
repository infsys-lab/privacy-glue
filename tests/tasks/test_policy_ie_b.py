#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from tasks.policy_ie_b import SUBTASKS, load_policy_ie_b


def test_load_policy_ie_b():
    # load sample data
    data = load_policy_ie_b(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "policy_ie_b")
    )

    # check that all three splits are included
    assert sorted(data.keys()) == sorted(["train", "validation", "test"])
    assert all([sorted(data[k].keys()) == SUBTASKS for k in data])

    # merge validation and train to train to compare against files
    for st in SUBTASKS:
        del data["validation"][st]

    # iterate over splits
    for (split, data_splits) in data.items():
        for (st, data_split) in data_splits.items():
            # check that all column names are as expected
            assert data_split.column_names == ["tokens", "tags", "subtask"]
            textspan = 1 + (SUBTASKS.index(st) * len(SUBTASKS))
            # define what is expected from the load function
            expected = [
                (
                    (f"{split}", "check", "for", "PolicyIE-B"),
                    ("O", "O", "O", f"label-{split}-{textspan}"),
                    st,
                ),
                (
                    ("another", f"{split}", "check", "for", "PolicyIE-B"),
                    ("O", "O", "O", "O", f"label-{split}-{textspan+1}"),
                    st,
                ),
            ]

            # assert that we got what is expected
            assert (
                list(
                    zip(
                        tuple(map(tuple, data_split["tokens"])),
                        tuple(map(tuple, data_split["tags"])),
                        data_split["subtask"],
                    )
                )
                == expected
            )
