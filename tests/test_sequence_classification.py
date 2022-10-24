#!/usr/bin/env python
# -*- coding: utf-8 -*-

from types import SimpleNamespace
from sequence_classification import Sequence_Classification_Pipeline
import datasets
import pytest


@pytest.mark.parametrize(
    "task, problem_type, input_keys",
    [
        ("opp_115", "multi_label", ["text"]),
        ("policy_detection", "single_label", ["text"]),
        ("policy_ie_a", "single_label", ["text"]),
        ("privacy_qa", "single_label", ["question", "text"]),
    ],
)
def test__init(task, problem_type, input_keys, mocked_arguments):
    # create mocked pipeline object
    mocked_pipeline = Sequence_Classification_Pipeline(*mocked_arguments(task=task))

    # make conditional assertion on problem type
    assert mocked_pipeline.problem_type == problem_type
    assert mocked_pipeline.input_keys == input_keys


@pytest.mark.parametrize(
    "problem_type",
    ["single_label", "multi_label"],
)
def test__retrieve_data(problem_type, mocked_arguments, mocker):
    # create mocked pipeline object
    mocked_pipeline = Sequence_Classification_Pipeline(
        *mocked_arguments(task="privacy_qa")
    )
    mocked_pipeline.problem_type = problem_type

    # mock relevant method
    get_data = mocker.patch(
        "sequence_classification.Sequence_Classification_Pipeline._get_data",
        return_value={
            "train": SimpleNamespace(
                features={"label": datasets.ClassLabel(names=["a", "b", "c"])}
            )
        }
        if problem_type == "single_label"
        else {
            "train": SimpleNamespace(
                features={
                    "label": datasets.Sequence(
                        datasets.ClassLabel(names=["d", "e", "f"])
                    )
                }
            )
        },
    )

    # execute relevant pipeline method
    mocked_pipeline._retrieve_data()

    # make assertions on changes
    get_data.assert_called_once()

    # make conditional assertions
    if problem_type == "single_label":
        assert mocked_pipeline.label_names == ["a", "b", "c"]
    else:
        assert mocked_pipeline.label_names == ["d", "e", "f"]


@pytest.mark.parametrize(
    "problem_type, problem_type_config",
    [
        ("single_label", "single_label_classification"),
        ("multi_label", "multi_label_classification"),
    ],
)
def test__load_pretrained_model_and_tokenizer(
    problem_type, problem_type_config, mocked_arguments, mocker
):
    # create mocked pipeline object
    current_arguments = mocked_arguments(task="privacy_qa")
    mocked_pipeline = Sequence_Classification_Pipeline(*current_arguments)
    mocked_pipeline.problem_type = problem_type
    mocked_pipeline.label_names = ["a", "b", "c"]

    # mock relevant modules
    auto_config = mocker.patch(
        "sequence_classification.AutoConfig.from_pretrained",
        return_value="mocked_config",
    )
    auto_tokenizer = mocker.patch(
        "sequence_classification.AutoTokenizer.from_pretrained",
        return_value="mocked_tokenizer",
    )
    auto_model = mocker.patch(
        "sequence_classification.AutoModelForSequenceClassification.from_pretrained",
        return_value="mocked_model",
    )

    # execute relevant pipeline method
    mocked_pipeline._load_pretrained_model_and_tokenizer()

    # make assertions
    auto_config.assert_called_once_with(
        current_arguments[1].model_name_or_path,
        cache_dir=current_arguments[1].cache_dir,
        revision=current_arguments[1].model_revision,
        problem_type=problem_type_config,
        num_labels=3,
        id2label={0: "a", 1: "b", 2: "c"},
        label2id={"a": 0, "b": 1, "c": 2},
    )
    auto_tokenizer.assert_called_once_with(
        current_arguments[1].model_name_or_path,
        cache_dir=current_arguments[1].cache_dir,
        use_fast=current_arguments[1].use_fast_tokenizer,
        revision=current_arguments[1].model_revision,
    )
    auto_model.assert_called_once_with(
        current_arguments[1].model_name_or_path,
        from_tf=mocker.ANY,
        config="mocked_config",
        cache_dir=current_arguments[1].cache_dir,
        revision=current_arguments[1].model_revision,
    )
    assert mocked_pipeline.config == "mocked_config"
    assert mocked_pipeline.tokenizer == "mocked_tokenizer"
    assert mocked_pipeline.model == "mocked_model"
