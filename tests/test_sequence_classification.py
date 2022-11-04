#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from collections import defaultdict
from types import SimpleNamespace

import datasets
import numpy as np
import pytest

from sequence_classification import Sequence_Classification_Pipeline


class Sequence_Classification_Pipeline_Override(Sequence_Classification_Pipeline):
    # override constant class variables
    task2problem = defaultdict(list)
    task2input = defaultdict(list)


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
    "task, problem_type, input_keys",
    [
        ("opp_115", "multi_label", ["text"]),
        ("policy_detection", "single_label", ["text"]),
        ("policy_ie_a", "single_label", ["text"]),
        ("privacy_qa", "single_label", ["question", "text"]),
    ],
)
def test__init__(task, problem_type, input_keys, mocked_arguments):
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
    mocked_pipeline = Sequence_Classification_Pipeline_Override(*mocked_arguments())
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
    current_arguments = mocked_arguments()
    mocked_pipeline = Sequence_Classification_Pipeline_Override(*current_arguments)
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


@pytest.mark.parametrize(
    "problem_type",
    ["single_label", "multi_label"],
)
@pytest.mark.parametrize(
    "do_train",
    [True, False],
)
@pytest.mark.parametrize(
    "do_eval",
    [True, False],
)
@pytest.mark.parametrize(
    "do_predict",
    [True, False],
)
@pytest.mark.parametrize(
    "max_seq_length",
    [512, 1024],
)
@pytest.mark.parametrize(
    "max_train_samples, actual_train_samples, "
    "max_eval_samples, actual_eval_samples, "
    "max_predict_samples, actual_predict_samples, "
    "overwrite_cache, preprocessing_num_workers",
    [
        (None, 3, None, 3, None, 3, True, None),
        (2, 2, 2, 2, 1, 1, True, 3),
        (12, 3, 11, 3, 10, 3, False, 2),
    ],
)
@pytest.mark.parametrize(
    "pad_to_max_length",
    [True, False],
)
@pytest.mark.parametrize(
    "fp16",
    [True, False],
)
def test__apply_preprocessing(
    problem_type,
    do_train,
    do_eval,
    do_predict,
    max_seq_length,
    max_train_samples,
    actual_train_samples,
    max_eval_samples,
    actual_eval_samples,
    max_predict_samples,
    actual_predict_samples,
    preprocessing_num_workers,
    overwrite_cache,
    pad_to_max_length,
    fp16,
    mocked_single_label_single_key_examples,
    mocked_arguments,
    mocker,
):
    # create mocked pipeline object
    mocked_pipeline = Sequence_Classification_Pipeline_Override(
        *mocked_arguments(
            max_train_samples=max_train_samples,
            max_eval_samples=max_eval_samples,
            max_predict_samples=max_predict_samples,
            max_seq_length=max_seq_length,
            preprocessing_num_workers=preprocessing_num_workers,
            overwrite_cache=overwrite_cache,
            pad_to_max_length=pad_to_max_length,
            do_train=do_train,
            do_eval=do_eval,
            do_predict=do_predict,
            fp16=fp16,
        )
    )
    mocked_pipeline.raw_datasets = mocked_single_label_single_key_examples
    mocked_pipeline.problem_type = problem_type
    mocked_pipeline.input_keys = ["text"]
    mocked_pipeline.label_names = ["a", "b", "c"]

    # define lots of mocks
    raw_datasets_train_methods = mocker.MagicMock()
    raw_datasets_eval_methods = mocker.MagicMock()
    raw_datasets_predict_methods = mocker.MagicMock()
    raw_datasets_train_methods.attach_mock(
        mocker.patch.object(
            mocked_pipeline.raw_datasets["train"],
            "select",
            return_value=mocked_pipeline.raw_datasets["train"],
        ),
        "select",
    )
    raw_datasets_train_methods.attach_mock(
        mocker.patch.object(
            mocked_pipeline.raw_datasets["train"],
            "map",
            return_value=mocked_pipeline.raw_datasets["train"],
        ),
        "map",
    )
    raw_datasets_eval_methods.attach_mock(
        mocker.patch.object(
            mocked_pipeline.raw_datasets["validation"],
            "select",
            return_value=mocked_pipeline.raw_datasets["validation"],
        ),
        "select",
    )
    raw_datasets_eval_methods.attach_mock(
        mocker.patch.object(
            mocked_pipeline.raw_datasets["validation"],
            "map",
            return_value=mocked_pipeline.raw_datasets["validation"],
        ),
        "map",
    )
    raw_datasets_predict_methods.attach_mock(
        mocker.patch.object(
            mocked_pipeline.raw_datasets["test"],
            "select",
            return_value=mocked_pipeline.raw_datasets["test"],
        ),
        "select",
    )
    raw_datasets_predict_methods.attach_mock(
        mocker.patch.object(
            mocked_pipeline.raw_datasets["test"],
            "map",
            return_value=mocked_pipeline.raw_datasets["test"],
        ),
        "map",
    )
    logger = mocker.patch.object(mocked_pipeline, "logger", create=True)
    tokenizer = mocker.patch.object(
        mocked_pipeline,
        "tokenizer",
        create=True,
        model_max_length=512,
    )
    mocker.patch(
        "sequence_classification.default_data_collator",
        return_value="default_data_collator",
    )
    data_collator_with_padding = mocker.patch(
        "sequence_classification.DataCollatorWithPadding",
        return_value="data_collator_with_padding",
    )

    # execute relevant pipeline method
    mocked_pipeline._apply_preprocessing()

    # make common assertions
    assert mocked_pipeline.max_seq_length == 512

    # make conditional assertions on sequence lengths
    if max_seq_length > tokenizer.model_max_length:
        logger.warning.assert_called_once()
    else:
        logger.warning.assert_not_called()

    # make conditional assertions on do_train
    if do_train:
        assert (
            mocked_pipeline.train_dataset
            == mocked_single_label_single_key_examples["train"]
        )
        if max_train_samples is not None:
            assert raw_datasets_train_methods.mock_calls == [
                mocker.call.select(range(actual_train_samples)),
                mocker.call.map(
                    mocker.ANY,
                    batched=True,
                    num_proc=preprocessing_num_workers,
                    load_from_cache_file=not overwrite_cache,
                    desc=mocker.ANY,
                    remove_columns=["text"]
                    if problem_type == "single_label"
                    else ["text", "label"],
                ),
            ]
        else:
            assert raw_datasets_train_methods.mock_calls == [
                mocker.call.map(
                    mocker.ANY,
                    batched=True,
                    num_proc=preprocessing_num_workers,
                    load_from_cache_file=not overwrite_cache,
                    desc=mocker.ANY,
                    remove_columns=["text"]
                    if problem_type == "single_label"
                    else ["text", "label"],
                ),
            ]
    else:
        assert not hasattr(mocked_pipeline, "train_dataset")
        assert raw_datasets_train_methods.mock_calls == []

    # make conditional assertions on do_eval
    if do_eval:
        assert (
            mocked_pipeline.eval_dataset
            == mocked_single_label_single_key_examples["validation"]
        )
        if max_eval_samples is not None:
            assert raw_datasets_eval_methods.mock_calls == [
                mocker.call.select(range(actual_eval_samples)),
                mocker.call.map(
                    mocker.ANY,
                    batched=True,
                    num_proc=preprocessing_num_workers,
                    load_from_cache_file=not overwrite_cache,
                    desc=mocker.ANY,
                    remove_columns=["text"]
                    if problem_type == "single_label"
                    else ["text", "label"],
                ),
            ]
        else:
            assert raw_datasets_eval_methods.mock_calls == [
                mocker.call.map(
                    mocker.ANY,
                    batched=True,
                    num_proc=preprocessing_num_workers,
                    load_from_cache_file=not overwrite_cache,
                    desc=mocker.ANY,
                    remove_columns=["text"]
                    if problem_type == "single_label"
                    else ["text", "label"],
                ),
            ]
    else:
        assert not hasattr(mocked_pipeline, "eval_dataset")
        assert raw_datasets_eval_methods.mock_calls == []

    # make conditional assertions on do_predict
    if do_predict:
        assert (
            mocked_pipeline.predict_dataset
            == mocked_single_label_single_key_examples["test"]
        )
        if max_predict_samples is not None:
            assert raw_datasets_predict_methods.mock_calls == [
                mocker.call.select(range(actual_predict_samples)),
                mocker.call.map(
                    mocker.ANY,
                    batched=True,
                    num_proc=preprocessing_num_workers,
                    load_from_cache_file=not overwrite_cache,
                    desc=mocker.ANY,
                    remove_columns=["text"]
                    if problem_type == "single_label"
                    else ["text", "label"],
                ),
            ]
        else:
            assert raw_datasets_predict_methods.mock_calls == [
                mocker.call.map(
                    mocker.ANY,
                    batched=True,
                    num_proc=preprocessing_num_workers,
                    load_from_cache_file=not overwrite_cache,
                    desc=mocker.ANY,
                    remove_columns=["text"]
                    if problem_type == "single_label"
                    else ["text", "label"],
                ),
            ]
    else:
        assert not hasattr(mocked_pipeline, "predict_dataset")
        assert raw_datasets_predict_methods.mock_calls == []

    # make conditional assertions on data collator
    if pad_to_max_length:
        assert mocked_pipeline.data_collator.return_value == "default_data_collator"
        data_collator_with_padding.assert_not_called()
    else:
        assert mocked_pipeline.data_collator == "data_collator_with_padding"
        data_collator_with_padding.assert_called_once()
        if fp16:
            assert (
                data_collator_with_padding.call_args.kwargs["pad_to_multiple_of"] == 8
            )
        else:
            assert (
                data_collator_with_padding.call_args.kwargs["pad_to_multiple_of"]
                is None
            )


@pytest.mark.parametrize(
    "problem_type, input_keys",
    [
        ("single_label", ["text"]),
        ("single_label", ["question", "text"]),
        ("multi_label", ["text"]),
    ],
)
@pytest.mark.parametrize(
    "pad_to_max_length",
    [True, False],
)
def test__preprocess_function(
    problem_type,
    input_keys,
    pad_to_max_length,
    mocked_single_label_single_key_examples,
    mocked_single_label_dual_key_examples,
    mocked_multi_label_single_key_examples,
    mocked_arguments,
    mocker,
):
    # create mocked pipeline object
    mocked_pipeline = Sequence_Classification_Pipeline_Override(
        *mocked_arguments(pad_to_max_length=pad_to_max_length)
    )
    mocked_pipeline.problem_type = problem_type
    mocked_pipeline.input_keys = input_keys
    mocked_pipeline.max_seq_length = 512
    mocked_pipeline.label_names = ["a", "b", "c"]

    # mock pipeline attributes
    tokenizer = mocker.patch.object(
        mocked_pipeline, "tokenizer", create=True, return_value={}
    )

    # execute relevant pipeline method
    if problem_type == "single_label" and len(input_keys) == 1:
        tokenized_examples = mocked_pipeline._preprocess_function(
            mocked_single_label_single_key_examples["train"]
        )
    elif problem_type == "single_label" and len(input_keys) == 2:
        tokenized_examples = mocked_pipeline._preprocess_function(
            mocked_single_label_dual_key_examples["train"]
        )
    else:
        tokenized_examples = mocked_pipeline._preprocess_function(
            mocked_multi_label_single_key_examples["train"]
        )

    # make conditional assertions
    if problem_type == "single_label" and len(input_keys) == 1:
        assert tokenized_examples == {}
        tokenizer.assert_called_once_with(
            [
                "train text for SC 1",
                "train text for SC 2",
                "train text for SC 3",
            ],
            padding="max_length" if pad_to_max_length else False,
            max_length=512,
            truncation=True,
        )
    elif problem_type == "single_label" and len(input_keys) == 2:
        assert tokenized_examples == {}
        tokenizer.assert_called_once_with(
            [
                "train question for SC 1",
                "train question for SC 2",
                "train question for SC 3",
            ],
            [
                "train text for SC 1",
                "train text for SC 2",
                "train text for SC 3",
            ],
            padding="max_length" if pad_to_max_length else False,
            max_length=512,
            truncation=True,
        )
    else:
        tokenizer.assert_called_once_with(
            [
                "train text for SC 1",
                "train text for SC 2",
                "train text for SC 3",
            ],
            padding="max_length" if pad_to_max_length else False,
            max_length=512,
            truncation=True,
        )
        assert list(tokenized_examples.keys()) == ["labels"]
        assert tokenized_examples["labels"] == [
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ]


@pytest.mark.parametrize(
    "problem_type, expected_metric_config",
    [("single_label", "multiclass"), ("multi_label", "multilabel")],
)
def test__set_metrics(problem_type, expected_metric_config, mocked_arguments, mocker):
    # create mocked pipeline object
    mocked_pipeline = Sequence_Classification_Pipeline_Override(*mocked_arguments())
    mocked_pipeline.problem_type = problem_type

    # create mocked objects
    evaluate = mocker.patch("sequence_classification.evaluate")
    evaluate.load.side_effect = lambda *args, **kwargs: args

    # execute relevant pipeline method
    mocked_pipeline._set_metrics()

    # make assertions
    assert mocked_pipeline.accuracy_metric == ("accuracy", expected_metric_config)
    assert mocked_pipeline.f1_metric == ("f1", expected_metric_config)
    assert mocked_pipeline.precision_metric == ("precision", expected_metric_config)
    assert mocked_pipeline.recall_metric == ("recall", expected_metric_config)
    assert mocked_pipeline.train_args.metric_for_best_model == "macro_f1"
    assert mocked_pipeline.train_args.greater_is_better


@pytest.mark.parametrize(
    "problem_type, scores, expected_predictions, label_ids",
    [
        (
            "single_label",
            [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]],
            np.array([0, 1]),
            np.array([1, 1]),
        ),
        (
            "multi_label",
            [[0.8, 0.8, -100], [-100, 0.8, 0.8]],
            np.array([[1, 1, 0], [0, 1, 1]]),
            np.array([[1, 0, 0], [0, 1, 0]]),
        ),
    ],
)
def test__compute_metrics(
    problem_type, scores, expected_predictions, label_ids, mocked_arguments, mocker
):
    # create mocked pipeline object
    mocked_pipeline = Sequence_Classification_Pipeline_Override(*mocked_arguments())
    mocked_pipeline.problem_type = problem_type
    f1_metric = mocker.patch.object(mocked_pipeline, "f1_metric", create=True)
    precision_metric = mocker.patch.object(
        mocked_pipeline, "precision_metric", create=True
    )
    recall_metric = mocker.patch.object(mocked_pipeline, "recall_metric", create=True)
    accuracy_metric = mocker.patch.object(
        mocked_pipeline, "accuracy_metric", create=True
    )

    # execute relevant pipeline method
    metric_dict = mocked_pipeline._compute_metrics(
        SimpleNamespace(predictions=scores, label_ids=label_ids)
    )

    # make assertions on calls
    for mocked_metric, metric in [
        (f1_metric, "f1"),
        (precision_metric, "precision"),
        (recall_metric, "recall"),
        (accuracy_metric, "accuracy"),
    ]:
        # cast all numpy arrays to lists for comparison
        for call in mocked_metric.mock_calls:
            for key, value in call.kwargs.items():
                if isinstance(value, np.ndarray):
                    call.kwargs[key] = value.tolist()

        # make conditional assertions
        if metric != "accuracy":
            assert mocked_metric.mock_calls == [
                mocker.call.compute(
                    predictions=expected_predictions.tolist(),
                    references=label_ids.tolist(),
                    average="macro",
                ),
                mocker.call.compute().__getitem__(metric),
                mocker.call.compute(
                    predictions=expected_predictions.tolist(),
                    references=label_ids.tolist(),
                    average="micro",
                ),
                mocker.call.compute().__getitem__(metric),
            ]
        else:
            assert mocked_metric.mock_calls == [
                mocker.call.compute(
                    predictions=expected_predictions.tolist(),
                    references=label_ids.tolist(),
                ),
                mocker.call.compute().__getitem__(metric),
            ]

    # make relevant assertions
    assert list(metric_dict.keys()) == [
        "macro_f1",
        "micro_f1",
        "macro_precision",
        "micro_precision",
        "macro_recall",
        "micro_recall",
        "accuracy",
    ]


@pytest.mark.parametrize(
    "local_rank",
    [-1, 0, 1],
)
@pytest.mark.parametrize(
    "do_train",
    [True, False],
)
@pytest.mark.parametrize(
    "do_eval",
    [True, False],
)
@pytest.mark.parametrize(
    "do_predict",
    [True, False],
)
@pytest.mark.parametrize(
    "early_stopping_patience",
    [None, 5],
)
@pytest.mark.parametrize(
    "task, problem_type",
    [
        ("opp_115", "multi_label"),
        ("policy_detection", "single_label"),
        ("policy_ie_a", "single_label"),
        ("privacy_qa", "single_label"),
    ],
)
def test__run_train_loop(
    local_rank,
    do_train,
    do_eval,
    do_predict,
    early_stopping_patience,
    task,
    problem_type,
    mocked_single_label_single_key_examples,
    mocked_single_label_dual_key_examples,
    mocked_multi_label_single_key_examples,
    mocked_arguments,
    mocker,
):
    # create mocked pipeline object
    current_arguments = mocked_arguments(
        task=task,
        local_rank=local_rank,
        do_train=do_train,
        do_eval=do_eval,
        do_predict=do_predict,
        early_stopping_patience=early_stopping_patience,
    )
    mocked_pipeline = Sequence_Classification_Pipeline(*current_arguments)
    mocked_pipeline.model = "model"
    mocked_pipeline.train_dataset = [1, 2, 3]
    mocked_pipeline.eval_dataset = [4, 5, 6]
    mocked_pipeline.predict_dataset = [7, 8, 9]
    mocked_pipeline.tokenizer = "tokenizer"
    mocked_pipeline.data_collator = "data_collator"
    mocked_pipeline._compute_metrics = "_compute_metrics"
    mocked_pipeline.last_checkpoint = "last_checkpoint"
    mocked_pipeline.label_names = ["a", "b", "c"]
    if task == "opp_115":
        mocked_pipeline.raw_datasets = mocked_multi_label_single_key_examples
    elif task == "privacy_qa":
        mocked_pipeline.raw_datasets = mocked_single_label_dual_key_examples
    else:
        mocked_pipeline.raw_datasets = mocked_single_label_single_key_examples

    # create mocked objects
    sc_trainer = mocker.patch(
        "sequence_classification.Trainer",
    )
    sc_trainer.return_value.configure_mock(
        **{
            "is_world_process_zero.return_value": current_arguments[2].local_rank
            in [-1, 0],
            "train.return_value": SimpleNamespace(metrics={}),
            "evaluate.return_value": {},
            "predict.return_value": (
                np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]),
                np.array([0, 0, 2]),
                {},
            )
            if problem_type == "single_label"
            else (
                np.array([[0.8, 0.8, -100.0], [-100.0, 0.8, 0.8], [0.8, -100.0, 0.8]]),
                np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0]]),
                {},
            ),
        }
    )
    wrs_trainer = mocker.patch(
        "sequence_classification.Weighted_Random_Sampler_Trainer",
    )
    wrs_trainer.return_value.configure_mock(
        **{
            "is_world_process_zero.return_value": current_arguments[2].local_rank
            in [-1, 0],
            "train.return_value": SimpleNamespace(metrics={}),
            "evaluate.return_value": {},
            "predict.return_value": (
                np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]),
                np.array([0, 0, 2]),
                {},
            )
            if problem_type == "single_label"
            else (
                np.array([[0.8, 0.8, -100.0], [-100.0, 0.8, 0.8], [0.8, -100.0, 0.8]]),
                np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0]]),
                {},
            ),
        }
    )
    early_stopping_callback = mocker.patch(
        "sequence_classification.EarlyStoppingCallback",
    )
    json_open_dump = mocker.MagicMock()
    json_open_dump.attach_mock(
        mocker.patch("sequence_classification.open", mocker.mock_open()), "open"
    )
    json_open_dump.attach_mock(
        mocker.patch("sequence_classification.json.dump"), "json_dump"
    )
    mocker.patch.object(mocked_pipeline, "logger", create=True)

    # execute relevant pipeline method
    mocked_pipeline._run_train_loop()

    # make conditional assertions for trainer
    if task == "privacy_qa":
        wrs_trainer.assert_called_once_with(
            model="model",
            args=mocker.ANY,
            train_dataset=[1, 2, 3] if do_train else None,
            eval_dataset=[4, 5, 6] if do_eval else None,
            tokenizer="tokenizer",
            data_collator="data_collator",
            compute_metrics="_compute_metrics",
            callbacks=[
                early_stopping_callback(early_stopping_patience=early_stopping_patience)
            ]
            if early_stopping_patience
            else None,
        )
        sc_trainer.assert_not_called()
    else:
        sc_trainer.assert_called_once_with(
            model="model",
            args=mocker.ANY,
            train_dataset=[1, 2, 3] if do_train else None,
            eval_dataset=[4, 5, 6] if do_eval else None,
            tokenizer="tokenizer",
            data_collator="data_collator",
            compute_metrics="_compute_metrics",
            callbacks=[
                early_stopping_callback(early_stopping_patience=early_stopping_patience)
            ]
            if early_stopping_patience
            else None,
        )
        wrs_trainer.assert_not_called()

    # make conditional assertions for training
    if do_train:
        mocked_pipeline.trainer.train.assert_called_once_with(
            resume_from_checkpoint="last_checkpoint"
        )
        mocked_pipeline.trainer.save_model.assert_called_once()
        mocked_pipeline.trainer.log_metrics.assert_any_call(
            "train", {"train_samples": 3}
        )
        mocked_pipeline.trainer.save_metrics.assert_any_call(
            "train", {"train_samples": 3}
        )
        mocked_pipeline.trainer.save_state.assert_called_once()
    else:
        mocked_pipeline.trainer.train.assert_not_called()
        mocked_pipeline.trainer.save_model.assert_not_called()
        mocked_pipeline.trainer.save_state.assert_not_called()
        with pytest.raises(AssertionError):
            mocked_pipeline.trainer.log_metrics.assert_any_call("train", mocker.ANY)
        with pytest.raises(AssertionError):
            mocked_pipeline.trainer.save_metrics.assert_any_call("train", mocker.ANY)

    # make conditional assertions for evaluation
    if do_eval:
        mocked_pipeline.trainer.evaluate.assert_called_once()
        mocked_pipeline.trainer.log_metrics.assert_any_call("eval", {"eval_samples": 3})
        mocked_pipeline.trainer.save_metrics.assert_any_call(
            "eval", {"eval_samples": 3}
        )
    else:
        mocked_pipeline.trainer.evaluate.assert_not_called()
        with pytest.raises(AssertionError):
            mocked_pipeline.trainer.log_metrics.assert_any_call("eval", mocker.ANY)
        with pytest.raises(AssertionError):
            mocked_pipeline.trainer.save_metrics.assert_any_call("eval", mocker.ANY)

    # make conditional assertions for prediction
    if do_predict:
        mocked_pipeline.trainer.predict.assert_called_once_with(
            [7, 8, 9], metric_key_prefix="predict"
        )
        mocked_pipeline.trainer.log_metrics.assert_any_call(
            "predict", {"predict_samples": 3}
        )
        mocked_pipeline.trainer.save_metrics.assert_any_call(
            "predict", {"predict_samples": 3}
        )
        if local_rank in [-1, 0]:
            # check that file was opened
            json_open_dump.open.assert_called_once_with(
                os.path.join(current_arguments[2].output_dir, "predictions.json"), "w"
            )

            # conditionally check dumped dictionary
            if task == "privacy_qa":
                json_open_dump.json_dump.assert_called_once_with(
                    [
                        {
                            "id": 0,
                            "question": "test question for SC 1",
                            "text": "test text for SC 1",
                            "gold_label": "a",
                            "predicted_label": "a",
                        },
                        {
                            "id": 1,
                            "question": "test question for SC 2",
                            "text": "test text for SC 2",
                            "gold_label": "a",
                            "predicted_label": "b",
                        },
                        {
                            "id": 2,
                            "question": "test question for SC 3",
                            "text": "test text for SC 3",
                            "gold_label": "c",
                            "predicted_label": "c",
                        },
                    ],
                    mocker.ANY,
                )
            elif task == "opp_115":
                json_open_dump.json_dump.assert_called_once_with(
                    [
                        {
                            "id": 0,
                            "text": "test text for SC 1",
                            "gold_label": ["a"],
                            "predicted_label": ["a", "b"],
                        },
                        {
                            "id": 1,
                            "text": "test text for SC 2",
                            "gold_label": ["b"],
                            "predicted_label": ["b", "c"],
                        },
                        {
                            "id": 2,
                            "text": "test text for SC 3",
                            "gold_label": ["b", "c"],
                            "predicted_label": ["a", "c"],
                        },
                    ],
                    mocker.ANY,
                )
            else:
                json_open_dump.json_dump.assert_called_once_with(
                    [
                        {
                            "id": 0,
                            "text": "test text for SC 1",
                            "gold_label": "a",
                            "predicted_label": "a",
                        },
                        {
                            "id": 1,
                            "text": "test text for SC 2",
                            "gold_label": "a",
                            "predicted_label": "b",
                        },
                        {
                            "id": 2,
                            "text": "test text for SC 3",
                            "gold_label": "c",
                            "predicted_label": "c",
                        },
                    ],
                    mocker.ANY,
                )
        else:
            json_open_dump.open.assert_not_called()
            json_open_dump.json_dump.assert_not_called()
    else:
        mocked_pipeline.trainer.predict.assert_not_called()
        with pytest.raises(AssertionError):
            mocked_pipeline.trainer.log_metrics.assert_any_call("predict", mocker.ANY)
        with pytest.raises(AssertionError):
            mocked_pipeline.trainer.save_metrics.assert_any_call("predict", mocker.ANY)
        json_open_dump.open.assert_not_called()
        json_open_dump.json_dump.assert_not_called()
