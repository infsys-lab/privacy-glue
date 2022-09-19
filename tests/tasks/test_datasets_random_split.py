#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datasets
import pytest


@pytest.mark.parametrize(
    "seed, expected",
    [
        (
            0,
            {
                "train": [5, 16, 8, 12, 0, 15, 17, 14, 20, 13, 9, 18, 22, 1],
                "validation": [3, 7, 21],
                "test": [19, 4, 10, 11, 24, 2, 23, 6],
            },
        ),
        (
            1,
            {
                "train": [6, 4, 14, 22, 12, 10, 5, 24, 8, 2, 19, 18, 9, 3],
                "validation": [11, 0, 13],
                "test": [1, 20, 7, 21, 17, 15, 23, 16],
            },
        ),
        (
            2,
            {
                "train": [17, 15, 9, 24, 4, 1, 3, 22, 5, 11, 19, 2, 0, 12],
                "validation": [10, 8, 13],
                "test": [14, 21, 20, 7, 16, 18, 23, 6],
            },
        ),
        (
            42,
            {
                "train": [12, 23, 14, 5, 0, 6, 13, 18, 3, 21, 1, 10, 22, 11],
                "validation": [4, 8, 2],
                "test": [15, 16, 19, 20, 9, 7, 17, 24],
            },
        ),
    ],
)
def test_datasets_random_split(seed, expected):
    # create an empty dataset dictionary
    combined = datasets.DatasetDict()

    # create splits via business-as-usual
    train_test_dataset_dict = datasets.Dataset.from_dict(
        {"text": range(25)}
    ).train_test_split(test_size=0.3, seed=seed)
    train_valid_dataset_dict = train_test_dataset_dict["train"].train_test_split(
        test_size=0.15, seed=seed
    )

    # assign splits to dataset dictionary
    combined["train"] = train_valid_dataset_dict["train"]
    combined["validation"] = train_valid_dataset_dict["test"]
    combined["test"] = train_test_dataset_dict["test"]

    # assert what is expected
    assert sorted(range(25)) == sorted(
        [index for _, data_split in combined.items() for index in data_split["text"]]
    )
    assert combined["train"]["text"] == expected["train"]
    assert combined["validation"]["text"] == expected["validation"]
    assert combined["test"]["text"] == expected["test"]
