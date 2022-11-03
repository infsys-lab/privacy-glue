#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from types import SimpleNamespace

import datasets
import pytest

from sequence_tagging import Sequence_Tagging_Pipeline

from torch import Tensor


@pytest.fixture
def mocked_examples():
    combined = datasets.DatasetDict()
    for split in ["train", "validation", "test"]:
        sample = {
            "tokens": [
                f"{split} text for SC 1",
                f"{split} text for SC 2",
                f"{split} text for SC 3",
            ],
            "tags": [
                ["O", "O", "O", "B-A", "I-A"],
                ["O", "O", "O", "B-A", "I-A"],
                ["O", "O", "O", "B-A", "I-A"],
            ],
            "subtask": ["task1", "task2", "task1"],
        }
        combined[split] = datasets.Dataset.from_dict(sample)

    return combined


@pytest.fixture
def mocked_pipeline(mocked_arguments):
    mocked_pipeline = Sequence_Tagging_Pipeline(*mocked_arguments())
    mocked_pipeline.subtasks = ["task1", "task2"]
    mocked_pipeline.label_names = {
        "task1": ["B-A", "I-A", "O"],
        "task2": ["B-B", "I-B", "O"],
    }
    return mocked_pipeline


mocked_prediction_vector = Tensor(
    [[[-1.0, 0.0, 1.0] for j in range(5)] for k in range(2)]
)


@pytest.fixture
def mocked_tokenized_inputs():
    class TokenizerOutput:
        def __init__(self, sample) -> None:
            self.ds = datasets.Dataset.from_dict(sample)

        def __call__(self) -> datasets.Dataset:
            return self.ds

        def word_ids(self, batch_index):
            return self.ds["word_ids"][batch_index]

    combined = datasets.DatasetDict()
    for split in ["train", "validation", "test"]:
        sample = {
            "word_ids": [
                [None, 0, 1, 2, 3, 4],
                [None, 0, 1, 2, 3, 4],
                [None, 0, 1, 2, 3, 4],
            ],
            "tags": [
                ["O", "O", "O", "B-A", "I-A.B"],
                ["O", "O", "O", "B-A", "I-A"],
                ["O", "O", "O", "B-A", "I-A"],
            ],
        }
        combined[split] = TokenizerOutput(sample)
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
    splits = ["train", "validation", "test"]
    # mock relevant method
    get_data = mocker.patch(
        "sequence_tagging.Sequence_Tagging_Pipeline._get_data",
        return_value={
            split: {
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
            for split in splits
        },
    )
    # patch sorted interleave function
    interleave = mocker.patch(
        "sequence_tagging.sorted_interleave_task_datasets",
        return_value=SimpleNamespace(features={"tags": datasets.Value("null")}),
    )

    # execute relevant pipeline method
    mocked_pipeline._retrieve_data()

    # make assertions on changes
    get_data.assert_called_once()
    interleave.assert_called_with(mocker.ANY, delete_features=True)

    # make conditional assertions
    for st in mocked_pipeline.subtasks:
        assert mocked_pipeline.label_names[st] == [f"{st}-a", f"{st}-b", f"{st}-c"]

    for split in splits:
        assert mocked_pipeline.raw_datasets[split].features["tags"] == datasets.Value(
            "null"
        )


@pytest.mark.parametrize(
    "task",
    ["piextract", "policy_ie_b"],
)
@pytest.mark.parametrize(
    "returned_model",
    [
        SimpleNamespace(__class__=SimpleNamespace(__name__="BERT")),
        SimpleNamespace(__class__=SimpleNamespace(__name__="Roberta")),
    ],
)
def test__load_pretrained_model_and_tokenizer(
    returned_model, task, mocked_arguments, mocker
):
    # create mocked pipeline object
    current_arguments = mocked_arguments(task=task)
    print(current_arguments)
    mocked_pipeline = Sequence_Tagging_Pipeline(*current_arguments)
    mocked_pipeline.label_names = {
        st: ["a", "b", "c"] for st in mocked_pipeline.subtasks
    }

    # mock relevant modules
    auto_config = mocker.patch(
        "sequence_tagging.AutoConfig.from_pretrained",
        return_value=returned_model,
    )
    auto_tokenizer = mocker.patch(
        "sequence_tagging.AutoTokenizer.from_pretrained",
        return_value="mocked_tokenizer",
    )
    auto_model = mocker.patch(
        "sequence_tagging.MultiTaskModel",
        return_value=returned_model,
    )

    # execute relevant pipeline method
    mocked_pipeline._load_pretrained_model_and_tokenizer()

    # make assertions
    auto_config.assert_called_once_with(
        current_arguments[1].model_name_or_path,
        cache_dir=current_arguments[1].cache_dir,
        revision=current_arguments[1].model_revision,
    )
    auto_tokenizer.assert_called_once_with(
        current_arguments[1].model_name_or_path,
        cache_dir=current_arguments[1].cache_dir,
        use_fast=True,
        revision=current_arguments[1].model_revision,
        add_prefix_space=True
        if returned_model.__class__.__name__.startswith("Roberta")
        else False,
    )
    auto_model.assert_called_once_with(
        current_arguments[1].model_name_or_path,
        tasks=mocked_pipeline.subtasks,
        config=returned_model,
        labels=mocked_pipeline.label_names,
    )
    assert mocked_pipeline.config == returned_model
    assert mocked_pipeline.tokenizer == "mocked_tokenizer"
    assert mocked_pipeline.model == returned_model


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
        (None, 3, None, 3, None, 3, True, 4),
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
    mocked_examples,
    mocked_arguments,
    mocker,
):
    # create mocked pipeline object
    mocked_pipeline = Sequence_Tagging_Pipeline(
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
    mocked_pipeline.subtasks = ["task1", "task2"]
    mocked_pipeline.raw_datasets = mocked_examples
    mocked_pipeline.label_names = {
        st: ["a", "b", "c"] for st in mocked_pipeline.subtasks
    }
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
        "sequence_tagging.default_data_collator",
        return_value="default_data_collator",
    )
    data_collator_with_padding = mocker.patch(
        "sequence_tagging.DataCollatorForTokenClassification",
        return_value="data_collator_for_token_classification",
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
        assert mocked_pipeline.train_dataset == mocked_examples["train"]
        if max_train_samples is not None:
            assert raw_datasets_train_methods.mock_calls == [
                mocker.call.select(range(actual_train_samples)),
                mocker.call.map(
                    mocker.ANY,
                    batched=True,
                    num_proc=preprocessing_num_workers,
                    load_from_cache_file=not overwrite_cache,
                    desc=mocker.ANY,
                    remove_columns=["tokens", "tags", "subtask"],
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
                    remove_columns=["tokens", "tags", "subtask"],
                ),
            ]
    else:
        assert not hasattr(mocked_pipeline, "train_dataset")
        assert raw_datasets_train_methods.mock_calls == []

    # make conditional assertions on do_eval
    if do_eval:
        assert mocked_pipeline.eval_dataset == mocked_examples["validation"]
        if max_eval_samples is not None:
            assert raw_datasets_eval_methods.mock_calls == [
                mocker.call.select(range(actual_eval_samples)),
                mocker.call.map(
                    mocker.ANY,
                    batched=True,
                    num_proc=preprocessing_num_workers,
                    load_from_cache_file=not overwrite_cache,
                    desc=mocker.ANY,
                    remove_columns=["tokens", "tags", "subtask"],
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
                    remove_columns=["tokens", "tags", "subtask"],
                ),
            ]
    else:
        assert not hasattr(mocked_pipeline, "eval_dataset")
        assert raw_datasets_eval_methods.mock_calls == []

    # make conditional assertions on do_predict
    if do_predict:
        assert mocked_pipeline.predict_dataset == mocked_examples["test"]
        if max_predict_samples is not None:
            assert raw_datasets_predict_methods.mock_calls == [
                mocker.call.select(range(actual_predict_samples)),
                mocker.call.map(
                    mocker.ANY,
                    batched=True,
                    num_proc=preprocessing_num_workers,
                    load_from_cache_file=not overwrite_cache,
                    desc=mocker.ANY,
                    remove_columns=["tokens", "tags", "subtask"],
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
                    remove_columns=["tokens", "tags", "subtask"],
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
        assert mocked_pipeline.data_collator == "data_collator_for_token_classification"
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
    "pad_to_max_length",
    [True, False],
)
def test__preprocess_function(
    pad_to_max_length,
    mocked_examples,
    mocked_arguments,
    mocker,
):
    # create mocked pipeline object
    mocked_pipeline = Sequence_Tagging_Pipeline(
        *mocked_arguments(pad_to_max_length=pad_to_max_length)
    )

    mocked_pipeline.max_seq_length = 512
    mocked_pipeline.label_names = {
        "task1": ["B-A", "I-A", "O"],
        "task2": ["B-B", "I-B", "O"],
    }

    # mock pipeline attributes
    tokenizer = mocker.patch.object(
        mocked_pipeline, "tokenizer", create=True, return_value={}
    )

    transformed = mocker.patch(
        "sequence_tagging.Sequence_Tagging_Pipeline._transform_labels_to_ids",
        return_value=("labels", "task_ids"),
    )
    # execute relevant pipeline method

    tokenized_examples = mocked_pipeline._preprocess_function(mocked_examples["train"])

    # make conditional assertions

    tokenizer.assert_called_once_with(
        [
            "train text for SC 1",
            "train text for SC 2",
            "train text for SC 3",
        ],
        padding="max_length" if pad_to_max_length else False,
        max_length=512,
        truncation=True,
        is_split_into_words=True,
    )
    transformed.assert_called_once()
    assert list(tokenized_examples.keys()) == ["labels", "task_ids"]
    assert tokenized_examples["labels"] == "labels"
    assert tokenized_examples["task_ids"] == "task_ids"


@pytest.mark.parametrize(
    "label_all_tokens",
    [True, False],
)
def test__transform_labels_to_ids(
    mocked_examples,
    mocked_tokenized_inputs,
    label_all_tokens,
    mocked_arguments,
    mocker,
):
    # create mocked pipeline object
    mocked_pipeline = Sequence_Tagging_Pipeline(
        *mocked_arguments(label_all_tokens=label_all_tokens)
    )
    mocked_pipeline.subtasks = ["task1", "task2"]
    mocked_pipeline.label_names = {
        "task1": ["B-A", "I-A", "O"],
        "task2": ["B-B", "I-B", "O"],
    }
    mocked_pipeline.b_to_i_label = mocked_pipeline._create_b_to_i_label_map()
    mocked_pipeline.label_to_ids = {"O": 0, "B-A": 1, "I-A": 2, "I-A.B": 3}
    labels, task_ids = mocked_pipeline._transform_labels_to_ids(
        mocked_examples["train"], mocked_tokenized_inputs["train"]
    )
    assert labels == [
        [-100, 0, 0, 0, 1, 2],
        [-100, 0, 0, 0, 1, 2],
        [-100, 0, 0, 0, 1, 2],
    ]
    assert task_ids == [0, 1, 0]


def test__create_b_to_i_label_map(mocked_pipeline):
    b2i = mocked_pipeline._create_b_to_i_label_map()
    assert b2i["task1"] == [1, 1, 2]


@pytest.mark.parametrize(
    "preds", [(mocked_prediction_vector,), mocked_prediction_vector]
)
def test__retransform_labels(preds, mocked_pipeline):
    labels = [[2] * 5] * 2
    true_p, true_l = mocked_pipeline._retransform_labels(preds, labels)
    print(true_p, true_l)
    assert true_p["task1"] == [["O", "O", "O", "O", "O"]]
    assert true_p["task2"] == [["O", "O", "O", "O", "O"]]
    assert true_p == true_l


def test__compute_metrics(mocked_pipeline):

    p = SimpleNamespace(predictions=mocked_prediction_vector, label_ids=[[2] * 5] * 2)
    metrics = mocked_pipeline.compute_metrics(p)
    assert metrics["task1_accuracy"] == 1.0
    assert metrics["task2_accuracy"] == 1.0
