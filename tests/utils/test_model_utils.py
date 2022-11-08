#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from types import SimpleNamespace

import pytest
import torch

from utils.model_utils import MultiTaskModel, TokenClassificationHead

LABEL_NAMES = {"task1": ["a", "b", "c"], "task2": ["a", "b"]}


def mocked_multi_task_arguments(tasks=["task1", "task2"], label_names=LABEL_NAMES):
    config = SimpleNamespace(hidden_size=5)
    return ("model_name", tasks, label_names, config)


@pytest.mark.parametrize(
    "tasks",
    [["task1", "task2"], ["task1"]],
)
def test_mtm_init(tasks, mocker):
    expected_config = SimpleNamespace(hidden_size=5)
    pretrained = mocker.patch(
        "utils.model_utils.AutoModel.from_pretrained",
        return_value=SimpleNamespace(config=expected_config),
    )
    mtm = MultiTaskModel(*mocked_multi_task_arguments(tasks=tasks))
    pretrained.assert_called_once_with("model_name", config=expected_config)
    assert len(mtm.output_heads) == len(tasks)
    assert set(mtm.output_heads.keys()) == set(map(str, range(len(tasks))))


@pytest.mark.parametrize("labels", [torch.CharTensor([0, 1, 1, 1]), None])
def test_mtm_forward(labels, mocker):
    expected_config = SimpleNamespace(hidden_size=5)
    mocker.patch(
        "utils.model_utils.AutoModel.from_pretrained",
        return_value=SimpleNamespace(config=expected_config),
    )
    tch_forward = mocker.patch(
        "utils.model_utils.TokenClassificationHead.forward",
        return_value=(torch.zeros(1, 2, 3), torch.zeros(1)),
    )
    mtm = MultiTaskModel(*mocked_multi_task_arguments())
    encoder = mocker.patch.object(
        mtm,
        "encoder",
        return_value=(torch.zeros(4, 2, 2), torch.zeros(4, 2, 2)),
    )
    output = mtm.forward(
        input_ids=torch.zeros(4),
        attention_mask=torch.CharTensor([1, 1, 1, 1]),
        labels=labels,
        task_ids=torch.CharTensor([0, 0, 0, 1]),
    )
    tch_forward.assert_called()
    encoder.assert_called()

    if labels is None:
        assert len(output) == 2
        assert output[0].shape == (2, 2, 3)
    else:
        assert output[1].shape == (2, 2, 3)
        assert output[0] == 0.0
        assert len(output) == 3


@pytest.mark.parametrize("num_labels", [1, 4])
def test_tch_init(num_labels):
    tch = TokenClassificationHead(4, num_labels)
    assert tch.num_labels == num_labels


@pytest.mark.parametrize("labels", [torch.CharTensor([[0, 1]] * 4), None])
@pytest.mark.parametrize("bias", [0.5, None])
@pytest.mark.parametrize("att_mask", [torch.CharTensor([[1, 1]] * 4), None])
def test_tch_forward(labels, bias, att_mask):
    tch = TokenClassificationHead(4, 2, bias=bias)
    output = tch.forward(
        sequence_output=torch.ones(8, 4),
        attention_mask=att_mask,
        labels=labels,
        pooled_output=0.0,
    )
    assert output[0].shape == (8, 2)
    if labels is not None:
        assert output[1] is not None
    else:
        assert output[1] is None
