#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import datasets
import pytest

from tasks.piextract import load_piextract, merge_ner_tags, read_conll_file


def mocked_conll_output():
    return {
        "tokens": [
            ["train", "check", "for", "PI-Extract"],
            ["another", "train", "check", "for", "PI-Extract"],
        ],
        "ner_tags": [
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
    assert sorted(data.keys()) == sorted(["tokens", "ner_tags"])

    # assert that we got what is expected
    assert data == mocked_conll_output()


@pytest.mark.parametrize(
    "raw_ner_tags, merged_ner_tags",
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
def test_merge_ner_tags(raw_ner_tags, merged_ner_tags):
    assert merge_ner_tags(raw_ner_tags) == merged_ner_tags


def test_load_piextract_mocked(mocker):
    # patch other units in the load_piextract function
    mocker.patch("tasks.piextract.read_conll_file", return_value=mocked_conll_output())
    mocker.patch(
        "tasks.piextract.merge_ner_tags", return_value=mocked_merge_tags_output()
    )

    # load sample data
    data = load_piextract(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "piextract")
    )

    # check that all three splits are included
    assert sorted(data.keys()) == sorted(["train", "validation", "test"])

    # merge validation and train to train to compare against files
    data["train"] = datasets.concatenate_datasets([data["train"], data["validation"]])
    del data["validation"]

    # iterate over splits
    for (split, data_split) in data.items():
        # check that all column names are as expected
        assert data_split.column_names == ["tokens", "ner_tags"]

        # assert that we got what is expected
        assert sorted(
            zip(
                map(tuple, data_split["tokens"]),
                [tuple(map(tuple, ner_tags)) for ner_tags in data_split["ner_tags"]],
            )
        ) == sorted(
            zip(
                map(tuple, mocked_conll_output()["tokens"]),
                [
                    tuple(map(tuple, ner_tags))
                    for ner_tags in mocked_merge_tags_output()
                ],
            )
        )


@pytest.mark.integration
def test_load_piextract():
    # load sample data
    data = load_piextract(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "piextract")
    )

    # check that all three splits are included
    assert sorted(data.keys()) == sorted(["train", "validation", "test"])

    # merge validation and train to train to compare against files
    data["train"] = datasets.concatenate_datasets([data["train"], data["validation"]])
    del data["validation"]

    # iterate over splits
    for (split, data_split) in data.items():
        # check that all column names are as expected
        assert data_split.column_names == ["tokens", "ner_tags"]

        # define what is expected from the load function
        expected = sorted(
            [
                (
                    (f"{split}", "check", "for", "PI-Extract"),
                    (
                        ("O", "O", "O", "O"),
                        ("O", "O", "O", "O"),
                        ("O", "O", "O", "O"),
                        (
                            f"label-{split}-collect-1",
                            f"label-{split}-nocollect-1",
                            f"label-{split}-share-1",
                            f"label-{split}-noshare-1",
                        ),
                    ),
                ),
                (
                    ("another", f"{split}", "check", "for", "PI-Extract"),
                    (
                        ("O", "O", "O", "O"),
                        ("O", "O", "O", "O"),
                        ("O", "O", "O", "O"),
                        ("O", "O", "O", "O"),
                        (
                            f"label-{split}-collect-2",
                            f"label-{split}-nocollect-2",
                            f"label-{split}-share-2",
                            f"label-{split}-noshare-2",
                        ),
                    ),
                ),
            ]
        )

        # assert that we got what is expected
        assert (
            sorted(
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
