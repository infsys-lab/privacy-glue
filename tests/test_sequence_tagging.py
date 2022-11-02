#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from types import SimpleNamespace

import datasets
import pytest

from sequence_tagging import Sequence_Tagging_Pipeline


@pytest.fixture
def mocked_single_label_single_key_examples():
    combined = datasets.DatasetDict()
    for split in ["train", "validation", "test"]:
        sample = {
            "text": [
                f"{split} text for SC 1",
                f"{split} text for SC 2",
                f"{split} text for SC 3",
            ],
            "label": [0, 1, 2],
        }
        combined[split] = datasets.Dataset.from_dict(sample)

    return combined


@pytest.fixture
def mocked_multi_label_single_key_examples():
    combined = datasets.DatasetDict()
    for split in ["train", "validation", "test"]:
        sample = {
            "text": [
                f"{split} text for SC 1",
                f"{split} text for SC 2",
                f"{split} text for SC 3",
            ],
            "label": [[0, 1], [1, 2], [0, 2]],
        }
        combined[split] = datasets.Dataset.from_dict(sample)

    return combined


@pytest.fixture
def mocked_single_label_dual_key_examples():
    combined = datasets.DatasetDict()
    for split in ["train", "validation", "test"]:
        sample = {
            "question": [
                f"{split} question for SC 1",
                f"{split} question for SC 2",
                f"{split} question for SC 3",
            ],
            "text": [
                f"{split} text for SC 1",
                f"{split} text for SC 2",
                f"{split} text for SC 3",
            ],
            "label": [0, 1, 2],
        }
        combined[split] = datasets.Dataset.from_dict(sample)

    return combined


@pytest.mark.parametrize(
    "task",
    [
        "piextract",
        "policy_ie_b",
        "opp-115",
    ],
)
def test__init__(task, mocked_arguments):
    # create mocked pipeline object
    mocked_pipeline = Sequence_Tagging_Pipeline(*mocked_arguments(task=task))

    # make conditional assertion on problem type
    if task == "piextract":
        assert mocked_pipeline.subtasks == [
            "COLLECT",
            "NOT_COLLECT",
            "NOT_SHARE",
            "SHARE",
        ]
    elif task == "policy_ie_b":
        assert mocked_pipeline.subtasks == ["type-I", "type-II"]
    else:
        assert mocked_pipeline.subtasks == [task]


@pytest.mark.parametrize(
    "task",
    ["piextract", "policy_ie_b"],
)
def test__retrieve_data(task, mocked_arguments, mocker):
    # create mocked pipeline object
    mocked_pipeline = Sequence_Tagging_Pipeline(*mocked_arguments(task=task))

    # mock relevant method
    get_data = mocker.patch(
        "sequence_tagging.Sequence_Tagging_Pipeline._get_data",
        return_value={
            "train": {
                st: SimpleNamespace(
                    features={
                        "tags": datasets.Sequence(
                            feature=datasets.ClassLabel(
                                names=[f"{st}-a", f"{st}-b", f"{st}-c"]
                            )
                        )
                    }
                )
                for st in mocked_pipeline.subtasks
            }
        },
    )

    interleave = mocker.patch(
        "utils.tasks_utils.sorted_interleave_task_datasets",
        return_value={
            "train": {
                st: SimpleNamespace(features={"tags": datasets.Value("null")})
                for st in mocked_pipeline.subtasks
            }
        },
    )

    # execute relevant pipeline method
    mocked_pipeline._retrieve_data()

    # make assertions on changes
    get_data.assert_called_once()
    interleave.assert_called_once()

    # make conditional assertions
    for st in mocked_pipeline.subtasks:
        assert mocked_pipeline.label_names[st] == [f"{st}-a", f"{st}-b", f"{st}-c"]

    for st in mocked_pipeline.subtasks:
        assert mocked_pipeline.raw_datasets["train"][st].features[
            "tags"
        ] == datasets.Value("null")
