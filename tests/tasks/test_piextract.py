#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tasks.piextract import load_piextract, merge_tags, read_conll_file, SUBTASKS
import datasets
import pytest
import os

TASK2TEXTSPAN = {
    "COLLECT": "collect",
    "NOT_COLLECT": "nocollect",
    "SHARE": "share",
    "NOT_SHARE": "noshare",
}


def mocked_conll_output():
    return {
        "tokens": [
            ["train", "check", "for", "PI-Extract"],
            ["another", "train", "check", "for", "PI-Extract"],
        ],
        "tags": [
            ["O", "O", "O", "label-train-nocollect-1"],
            ["O", "O", "O", "O", "label-train-nocollect-2"],
        ],
    }


def mocked_merge_tags_output():
    return [
        [
            ["O", "O", "O", "O"],
            ["O", "O", "O", "O"],
            ["O", "O", "O", "O"],
            [
                "label-train-nocollect-1",
                "label-train-nocollect-1",
                "label-train-nocollect-1",
                "label-train-nocollect-1",
            ],
        ],
        [
            ["O", "O", "O", "O"],
            ["O", "O", "O", "O"],
            ["O", "O", "O", "O"],
            ["O", "O", "O", "O"],
            [
                "label-train-nocollect-2",
                "label-train-nocollect-2",
                "label-train-nocollect-2",
                "label-train-nocollect-2",
            ],
        ],
    ]


def mocked_expand_dataset_per_task():
    return {
        st: [
            (
                ("train", "check", "for", "PI-Extract"),
                ("O", "O", "O", "label-train-nocollect-1"),
                st,
            ),
            (
                ("another", "train", "check", "for", "PI-Extract"),
                ("O", "O", "O", "O", "label-train-nocollect-2"),
                st,
            ),
        ]
        for st in SUBTASKS
    }


def test_read_conll_file():
    conll_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "piextract",
        "CollectUse_false",
        "train.conll03",
    )

    # read CONLL file
    data = read_conll_file(conll_file)

    # check their keys make sense
    assert sorted(data.keys()) == sorted(["tokens", "tags"])

    # assert that we got what is expected
    assert data == mocked_conll_output()


@pytest.mark.parametrize(
    "raw_tags, merged_tags",
    [
        (
            [
                [["O", "O", "B-Thing", "I-Thing", "O"], ["O", "O"]],
                [["O", "B-Thang", "I-Thang", "O", "O"], ["B-Thing", "I-Thing"]],
            ],
            [
                [
                    ("O", "O"),
                    ("O", "B-Thang"),
                    ("B-Thing", "I-Thang"),
                    ("I-Thing", "O"),
                    ("O", "O"),
                ],
                [("O", "B-Thing"), ("O", "I-Thing")],
            ],
        )
    ],
)
def test_merge_tags(raw_tags, merged_tags):
    assert merge_tags(raw_tags) == merged_tags


def test_load_piextract_mocked(mocker):
    # patch other units in the load_piextract function
    mocker.patch("tasks.piextract.read_conll_file", return_value=mocked_conll_output())
    mocker.patch("tasks.piextract.merge_tags", return_value=mocked_merge_tags_output())
    mocker.patch(
        "utils.tasks_utils.expand_dataset_per_task",
        return_value=mocked_expand_dataset_per_task(),
    )

    # load sample data
    data = load_piextract(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "piextract")
    )

    # check that all three splits are included
    assert sorted(data.keys()) == sorted(["train", "validation", "test"])

    # merge validation and train to train to compare against files
    for st in SUBTASKS:
        data["train"][st] = datasets.concatenate_datasets(
            [data["train"][st], data["validation"][st]]
        )
        del data["validation"][st]

    # iterate over splits
    for (split, data_splits) in data.items():
        for (st, data_split) in sorted(data_splits.items()):
            # check that all column names are as expected
            assert data_split.column_names == ["tokens", "tags", "subtask"]
            # assert that we got what is expected
            assert (
                list(
                    zip(
                        [tuple(tokens) for tokens in data_split["tokens"]],
                        [tuple(tags) for tags in data_split["tags"]],
                        (st, st),
                    )
                )
                == mocked_expand_dataset_per_task()[st]
            )


@pytest.mark.integration
def test_load_piextract():
    # load sample data
    data = load_piextract(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "piextract")
    )

    # check that all three splits are included
    assert sorted(data.keys()) == sorted(["train", "validation", "test"])
    assert all([sorted(data[k].keys()) == SUBTASKS for k in data])

    # merge validation and train to train to compare against files
    for st in SUBTASKS:
        data["train"][st] = datasets.concatenate_datasets(
            [data["train"][st], data["validation"][st]]
        )
        del data["validation"][st]

    # iterate over splits
    for (split, data_splits) in data.items():
        for (st, data_split) in data_splits.items():
            # check that all column names are as expected
            assert data_split.column_names == ["tokens", "tags", "subtask"]
            # define what is expected from the load function
            expected = [
                (
                    (f"{split}", "check", "for", "PI-Extract"),
                    ("O", "O", "O", f"label-{split}-{TASK2TEXTSPAN[st]}-1"),
                    st,
                ),
                (
                    ("another", f"{split}", "check", "for", "PI-Extract"),
                    ("O", "O", "O", "O", f"label-{split}-{TASK2TEXTSPAN[st]}-2"),
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
