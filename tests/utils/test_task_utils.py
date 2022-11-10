#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datasets
import pytest

from utils.task_utils import expand_dataset_per_task, sorted_interleave_task_datasets


def make_task_ds(tokens, tags, task):
    return datasets.Dataset.from_dict({"tokens": tokens, "tags": tags, "subtask": task})


@pytest.mark.parametrize("delete_features", [True, False])
def test_sorted_interleave_task_datasets(delete_features):
    tasks = ["task1", "task2", "task3"]
    input = {
        "task1": make_task_ds([["a", "b", "c"]] * 2, [[0, 1, 0]] * 2, ["task1"] * 2),
        "task2": make_task_ds([["a", "b", "c"]] * 2, [[1, 2, 0]] * 2, ["task2"] * 2),
        "task3": make_task_ds([["a", "b", "c"]] * 2, [[0, 0, 0]] * 2, ["task3"] * 2),
    }
    actual = sorted_interleave_task_datasets(input, delete_features)

    expected = datasets.Dataset.from_dict(
        {
            "tokens": [input[task]["tokens"][i] for i in range(2) for task in tasks],
            "tags": [input[task]["tags"][i] for i in range(2) for task in tasks],
            "subtask": [input[task]["subtask"][i] for i in range(2) for task in tasks],
        }
    )
    assert actual["tokens"] == expected["tokens"]
    assert actual["tags"] == expected["tags"]
    assert actual["subtask"] == expected["subtask"]


def test_expand_dataset_per_task():
    tasks = ["task1", "task2", "task3"]
    example_ds = datasets.Dataset.from_dict(
        {
            "tokens": [["a", "b", "c"]] * 2,
            "tags": [[[0, 1, 0], [1, 2, 0], [0, 0, 0]]] * 2,
        }
    )
    output_per_task = expand_dataset_per_task(example_ds, tasks)

    expected = {
        "task1": make_task_ds([["a", "b", "c"]] * 2, [[0, 1, 0]] * 2, ["task1"] * 2),
        "task2": make_task_ds([["a", "b", "c"]] * 2, [[1, 2, 0]] * 2, ["task2"] * 2),
        "task3": make_task_ds([["a", "b", "c"]] * 2, [[0, 0, 0]] * 2, ["task3"] * 2),
    }
    for task in tasks:
        assert output_per_task[task]["tokens"] == expected[task]["tokens"]
        assert output_per_task[task]["tags"] == expected[task]["tags"]
        assert output_per_task[task]["subtask"] == expected[task]["subtask"]
