#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tempfile

import pytest

from parser import (
    DataArguments,
    ExperimentArguments,
    ModelArguments,
    TrainingArguments,
    get_parser,
)


@pytest.mark.parametrize(
    "model_name_or_path",
    [
        "bert-base-uncased",
        "roberta-base",
        "nlpaueb/legal-bert-base-uncased",
        "saibo/legal-roberta-base",
        "mukund/privbert",
    ],
)
def test_ModelArguments_valid_model(model_name_or_path):
    # no error on known models
    model_args = ModelArguments(model_name_or_path=model_name_or_path)
    assert model_args.model_revision is not None
    assert model_args.model_revision != "main"


@pytest.mark.parametrize(
    "model_name_or_path",
    [
        "bert-base-uncased",
        "roberta-base",
        "nlpaueb/legal-bert-base-uncased",
        "saibo/legal-roberta-base",
        "mukund/privbert",
    ],
)
def test_ModelArguments_specific_revision(model_name_or_path):
    # model revision should be propagated
    model_args = ModelArguments(
        model_name_or_path=model_name_or_path, model_revision="main"
    )
    assert model_args.model_revision == "main"


@pytest.mark.parametrize(
    "model_name_or_path",
    ["bert", "roberta"],
)
def test_ModelArguments_invalid_model(model_name_or_path):
    # error on unknown models
    with pytest.raises(AssertionError):
        ModelArguments(model_name_or_path=model_name_or_path)


@pytest.mark.parametrize(
    "task",
    [
        "opp_115",
        "piextract",
        "policy_detection",
        "policy_ie_a",
        "policy_ie_b",
        "policy_qa",
        "privacy_qa",
        "all",
    ],
)
def test_DataArguments_valid_task(task):
    # no error on known task
    DataArguments(task=task)


@pytest.mark.parametrize(
    "task",
    ["privacy_ie", "policy_ie"],
)
def test_DataArguments_invalid_task(task):
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
