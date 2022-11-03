#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tempfile
from parser import (
    DataArguments,
    ExperimentArguments,
    ModelArguments,
    TrainingArguments,
    get_parser,
)

import pytest


@pytest.mark.parametrize(
    "model_name_or_path",
    ["bert-base-uncased", "roberta-base"],
)
def test_ModelArguments_valid(model_name_or_path):
    # no error on known models
    ModelArguments(model_name_or_path=model_name_or_path)


@pytest.mark.parametrize(
    "model_name_or_path",
    ["bert", "roberta"],
)
def test_ModelArguments_invalid(model_name_or_path):
    # error on unknown models
    with pytest.raises(AssertionError):
        ModelArguments(model_name_or_path=model_name_or_path)


@pytest.mark.parametrize(
    "task",
    ["privacy_qa", "policy_qa"],
)
def test_DataArguments_task_valid(task):
    # no error on known task
    DataArguments(task=task)


@pytest.mark.parametrize(
    "task",
    ["privacy_ie", "policy_ie"],
)
def test_DataArguments_task_invalid(task):
    # error on unknown task
    with pytest.raises(AssertionError):
        DataArguments(task=task)


def test_DataArguments_data_dir():
    # no error on valid directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        DataArguments(task="all", data_dir=tmp_dir)

    # create and delete a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        pass

    # error on invalid directory
    with pytest.raises(AssertionError):
        DataArguments(task="all", data_dir=tmp_dir)


def test_get_parser():
    # get parser and check contents
    parser = get_parser()
    assert parser.dataclass_types == [
        DataArguments,
        ModelArguments,
        TrainingArguments,
        ExperimentArguments,
    ]
