#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from parser import ModelArguments, DataArguments, TrainingArguments, get_parser
import tempfile
import pytest


@pytest.mark.parametrize(
    "model_valid, model_invalid",
    [("bert-base-uncased", "bert"), ("all-mpnet-base-v2", "all-mpnet")],
)
def test_ModelArguments(model_valid, model_invalid):
    # no error on known models
    try:
        ModelArguments(model_name_or_path=model_valid)
    except AssertionError:
        pytest.fail("Unexpected assertion encountered")

    # error on unknown models
    with pytest.raises(Exception) as exception_info:
        ModelArguments(model_name_or_path=model_invalid)
    assert exception_info.type == AssertionError


@pytest.mark.parametrize(
    "task_valid, task_invalid",
    [("policy_qa", "policy_extract"), ("privacy_qa", "privacy_ie")],
)
def test_DataArguments(task_valid, task_invalid):
    # no error on known task
    try:
        DataArguments(task=task_valid)
    except AssertionError:
        pytest.fail("Unexpected assertion encountered")

    # no error on valid directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            DataArguments(task="all", data_dir=tmp_dir)
        except AssertionError:
            pytest.fail("Unexpected assertion encountered")

    # error on unknown task
    with pytest.raises(Exception) as exception_info:
        DataArguments(task=task_invalid)
    assert exception_info.type == AssertionError

    # create and delete a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        pass

    # error on invalid directory
    with pytest.raises(Exception) as exception_info:
        DataArguments(task="all", data_dir=tmp_dir)
    assert exception_info.type == AssertionError


def test_get_parser():
    # get parser and check contents
    parser = get_parser()
    assert set(parser.dataclass_types) == {
        ModelArguments,
        DataArguments,
        TrainingArguments,
    }
