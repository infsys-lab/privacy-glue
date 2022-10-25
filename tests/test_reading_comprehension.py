#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from types import SimpleNamespace
from datasets.arrow_dataset import Batch
from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    AutoTokenizer,
)
from reading_comprehension import Reading_Comprehension_Pipeline
import numpy as np
import datasets
import pytest


@pytest.fixture
def mocked_qa_examples_features_predictions():
    # define configuration in case it is useful
    config = {"max_seq_length": 18, "doc_stride": 4}

    # define hard-coded examples
    examples = datasets.Dataset.from_dict(
        {
            "id": [f"sample_id_{index}" for index in range(1, 4)],
            "title": ["sample.com"] * 3,
            "context": [
                "sample answer for PolicyQA",
                "another sample answer for PolicyQA",
                "yet another sample answer for PolicyQA with extra text",
            ],
            "question": [
                "sample question for PolicyQA 1?",
                "sample question for PolicyQA 2?",
                "sample question for PolicyQA 3?",
            ],
            "answers": [
                {
                    "text": ["sample", "answer for PolicyQA"],
                    "answer_start": [0, 7],
                },
                {"text": ["another", "sample"], "answer_start": [0, 8]},
                {"text": ["answer"], "answer_start": [19]},
            ],
        }
    )

    # define hard-coded features
    features = datasets.Dataset.from_dict(
        {
            "input_ids": [
                [
                    101,
                    7099,
                    3160,
                    2005,
                    3343,
                    19062,
                    1015,
                    1029,
                    102,
                    7099,
                    3437,
                    2005,
                    3343,
                    19062,
                    102,
                    0,
                    0,
                    0,
                ],
                [
                    101,
                    7099,
                    3160,
                    2005,
                    3343,
                    19062,
                    1016,
                    1029,
                    102,
                    2178,
                    7099,
                    3437,
                    2005,
                    3343,
                    19062,
                    102,
                    0,
                    0,
                ],
                [
                    101,
                    7099,
                    3160,
                    2005,
                    3343,
                    19062,
                    1017,
                    1029,
                    102,
                    2664,
                    2178,
                    7099,
                    3437,
                    2005,
                    3343,
                    19062,
                    2007,
                    102,
                ],
                [
                    101,
                    7099,
                    3160,
                    2005,
                    3343,
                    19062,
                    1017,
                    1029,
                    102,
                    2005,
                    3343,
                    19062,
                    2007,
                    4469,
                    3793,
                    102,
                    0,
                    0,
                ],
            ],
            "token_type_ids": [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            ],
            "attention_mask": [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            ],
            "offset_mapping": [
                [
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    (0, 6),
                    (7, 13),
                    (14, 17),
                    (18, 24),
                    (24, 26),
                    None,
                    None,
                    None,
                    None,
                ],
                [
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    (0, 7),
                    (8, 14),
                    (15, 21),
                    (22, 25),
                    (26, 32),
                    (32, 34),
                    None,
                    None,
                    None,
                ],
                [
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    (0, 3),
                    (4, 11),
                    (12, 18),
                    (19, 25),
                    (26, 29),
                    (30, 36),
                    (36, 38),
                    (39, 43),
                    None,
                ],
                [
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    (26, 29),
                    (30, 36),
                    (36, 38),
                    (39, 43),
                    (44, 49),
                    (50, 54),
                    None,
                    None,
                    None,
                ],
            ],
            "example_id": ["sample_id_1", "sample_id_2", "sample_id_3", "sample_id_3"],
        }
    )

    # define hard-coded predictions
    predictions = (
        np.array(
            [
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.9,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.1,
                    0.1,
                    0.1,
                    0.9,
                    0.1,
                    0.1,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.1,
                    0.1,
                    0.9,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    1.0,
                    0.1,
                    0.0,
                    0.0,
                    0.0,
                ],
            ]
        ),
        np.array(
            [
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.9,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.9,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.9,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            ]
        ),
    )

    return examples, features, predictions, config


@pytest.fixture
def mocked_qa_examples():
    combined = datasets.DatasetDict()

    for split in ["train", "validation", "test"]:
        sample = {
            "id": [f"{split}_id_{index}" for index in range(1, 9)],
            "title": [f"{split}.com"] * 8,
            "context": [
                f"{split} answer for PolicyQA",
                f"{split} answer for PolicyQA",
                f"another {split} answer for PolicyQA",
                f"empty {split} answer for PolicyQA",
                " ".join(["huge sample answer for PolicyQA"] * 110),
                " ".join(["huge sample answer for PolicyQA"] * 110),
                " ".join(["huge sample answer for PolicyQA"] * 110),
                " ".join(["huge sample answer for PolicyQA"] * 110),
            ],
            "question": [
                f"{split} question for PolicyQA 1?",
                f"{split} question for PolicyQA 2?",
                f"{split} question for PolicyQA 3?",
                f"{split} question for PolicyQA 4?",
                f"{split} question for PolicyQA 5?",
                f"{split} question for PolicyQA 6?",
                f"{split} question for PolicyQA 7?",
                f"{split} question for PolicyQA 8?",
            ],
            "answers": [
                {
                    "text": [f"{split} answer", "answer for PolicyQA"],
                    "answer_start": [0, len(split) + 1],
                },
                {"text": [f"{split}", "answer"], "answer_start": [0, len(split) + 1]},
                {"text": [f"{split} answer"], "answer_start": [8]},
                {"text": [], "answer_start": []},
                {
                    "text": [" ".join(["huge sample answer for PolicyQA"] * 22)],
                    "answer_start": [1984],
                },
                {
                    "text": ["huge sample answer for PolicyQA"],
                    "answer_start": [1984],
                },
                {
                    "text": ["huge sample answer for PolicyQA"],
                    "answer_start": [2688],
                },
                {
                    "text": ["huge sample answer for PolicyQA"],
                    "answer_start": [2016],
                },
            ],
        }
        combined[split] = datasets.Dataset.from_dict(sample)

    return combined


def test__retrieve_data(mocked_arguments, mocker):
    # create mocked pipeline object
    mocked_pipeline = Reading_Comprehension_Pipeline(*mocked_arguments())

    # mock relevant method
    get_data = mocker.patch(
        "reading_comprehension.Reading_Comprehension_Pipeline._get_data",
        return_value="mocked_dictionary",
    )

    # execute relevant pipeline method
    mocked_pipeline._retrieve_data()

    # make assertions on changes
    get_data.assert_called_once()
    assert mocked_pipeline.raw_datasets == "mocked_dictionary"


@pytest.mark.parametrize(
    "use_fast",
    [True, False],
)
def test__load_pretrained_model_and_tokenizer(use_fast, mocked_arguments, mocker):
    # create mocked pipeline object
    current_arguments = mocked_arguments()
    mocked_pipeline = Reading_Comprehension_Pipeline(*current_arguments)

    # mock relevant modules
    auto_config = mocker.patch(
        "reading_comprehension.AutoConfig.from_pretrained",
        return_value="mocked_config",
    )
    auto_tokenizer = mocker.patch(
        "reading_comprehension.AutoTokenizer.from_pretrained",
        return_value=mocker.MagicMock(
            spec=PreTrainedTokenizerFast if use_fast else PreTrainedTokenizer,
            return_value="mocked_tokenizer",
        ),
    )
    auto_model = mocker.patch(
        "reading_comprehension.AutoModelForQuestionAnswering.from_pretrained",
        return_value="mocked_model",
    )

    # execute relevant pipeline method
    if use_fast:
        mocked_pipeline._load_pretrained_model_and_tokenizer()
    else:
        with pytest.raises(ValueError):
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
    )
    auto_model.assert_called_once_with(
        current_arguments[1].model_name_or_path,
        from_tf=mocker.ANY,
        config="mocked_config",
        cache_dir=current_arguments[1].cache_dir,
        revision=current_arguments[1].model_revision,
    )
    assert mocked_pipeline.config == "mocked_config"
    assert mocked_pipeline.tokenizer.return_value == "mocked_tokenizer"
    assert mocked_pipeline.model == "mocked_model"


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
        (None, 8, None, 8, None, 8, True, 4),
        (2, 2, 2, 2, 1, 1, True, 3),
        (12, 8, 11, 8, 10, 8, False, 2),
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
    do_train,
    do_eval,
    do_predict,
    fp16,
    mocked_qa_examples,
    mocked_arguments,
    mocker,
):
    # create mocked pipeline object
    mocked_pipeline = Reading_Comprehension_Pipeline(
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
    mocked_pipeline.raw_datasets = mocked_qa_examples

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
        padding_side="right",
        model_max_length=512,
    )
    mocker.patch(
        "reading_comprehension.default_data_collator",
        return_value="default_data_collator",
    )
    data_collator_with_padding = mocker.patch(
        "reading_comprehension.DataCollatorWithPadding",
        return_value="data_collator_with_padding",
    )

    # execute relevant pipeline method
    mocked_pipeline._apply_preprocessing()

    # make common assertions
    assert mocked_pipeline.question_column_name == "question"
    assert mocked_pipeline.context_column_name == "context"
    assert mocked_pipeline.answer_column_name == "answers"
    assert mocked_pipeline.pad_on_right
    assert mocked_pipeline.max_seq_length == 512

    # make conditional assertions on sequence lengths
    if max_seq_length > tokenizer.model_max_length:
        logger.warning.assert_called_once()
    else:
        logger.warning.assert_not_called()

    # make conditional assertions on do_train
    if do_train:
        assert mocked_pipeline.train_dataset == mocked_qa_examples["train"]
        if max_train_samples is not None:
            assert raw_datasets_train_methods.mock_calls == [
                mocker.call.select(range(actual_train_samples)),
                mocker.call.map(
                    mocker.ANY,
                    batched=True,
                    num_proc=preprocessing_num_workers,
                    remove_columns=["id", "title", "context", "question", "answers"],
                    load_from_cache_file=not overwrite_cache,
                    desc=mocker.ANY,
                ),
                mocker.call.select(range(actual_train_samples)),
            ]
        else:
            assert raw_datasets_train_methods.mock_calls == [
                mocker.call.map(
                    mocker.ANY,
                    batched=True,
                    num_proc=preprocessing_num_workers,
                    remove_columns=["id", "title", "context", "question", "answers"],
                    load_from_cache_file=not overwrite_cache,
                    desc=mocker.ANY,
                ),
            ]
    else:
        assert not hasattr(mocked_pipeline, "train_dataset")
        assert raw_datasets_train_methods.mock_calls == []

    # make conditional assertions on do_eval
    if do_eval:
        assert mocked_pipeline.eval_dataset == mocked_qa_examples["validation"]
        if max_eval_samples is not None:
            assert raw_datasets_eval_methods.mock_calls == [
                mocker.call.select(range(actual_eval_samples)),
                mocker.call.map(
                    mocker.ANY,
                    batched=True,
                    num_proc=preprocessing_num_workers,
                    remove_columns=["id", "title", "context", "question", "answers"],
                    load_from_cache_file=not overwrite_cache,
                    desc=mocker.ANY,
                ),
                mocker.call.select(range(actual_eval_samples)),
            ]
        else:
            assert raw_datasets_eval_methods.mock_calls == [
                mocker.call.map(
                    mocker.ANY,
                    batched=True,
                    num_proc=preprocessing_num_workers,
                    remove_columns=["id", "title", "context", "question", "answers"],
                    load_from_cache_file=not overwrite_cache,
                    desc=mocker.ANY,
                ),
            ]
    else:
        assert not hasattr(mocked_pipeline, "eval_dataset")
        assert raw_datasets_eval_methods.mock_calls == []

    # make conditional assertions on do_predict
    if do_predict:
        assert mocked_pipeline.predict_dataset == mocked_qa_examples["test"]
        if max_predict_samples is not None:
            assert raw_datasets_predict_methods.mock_calls == [
                mocker.call.select(range(actual_predict_samples)),
                mocker.call.map(
                    mocker.ANY,
                    batched=True,
                    num_proc=preprocessing_num_workers,
                    remove_columns=["id", "title", "context", "question", "answers"],
                    load_from_cache_file=not overwrite_cache,
                    desc=mocker.ANY,
                ),
                mocker.call.select(range(actual_predict_samples)),
            ]
        else:
            assert raw_datasets_predict_methods.mock_calls == [
                mocker.call.map(
                    mocker.ANY,
                    batched=True,
                    num_proc=preprocessing_num_workers,
                    remove_columns=["id", "title", "context", "question", "answers"],
                    load_from_cache_file=not overwrite_cache,
                    desc=mocker.ANY,
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


def test__prepare_split_features(mocked_arguments, mocked_qa_examples):
    # create mocked pipeline object
    mocked_pipeline = Reading_Comprehension_Pipeline(*mocked_arguments())
    mocked_pipeline.raw_datasets = mocked_qa_examples
    mocked_pipeline.question_column_name = "question"
    mocked_pipeline.context_column_name = "context"
    mocked_pipeline.answer_column_name = "answers"
    mocked_pipeline.pad_on_right = True
    mocked_pipeline.max_seq_length = 512

    # define inner private tokenizer function
    def _get_tokenizer(local_files_only):
        return AutoTokenizer.from_pretrained(
            "bert-base-uncased",
            use_fast=True,
            revision="5546055f03398095e385d7dc625e636cc8910bf2",
            local_files_only=local_files_only,
        )

    # load tokenizer
    try:
        mocked_pipeline.tokenizer = _get_tokenizer(local_files_only=True)
    except OSError:
        mocked_pipeline.tokenizer = _get_tokenizer(local_files_only=False)

    # execute relevant pipeline method
    train_features = mocked_pipeline._prepare_train_features(
        Batch(mocked_pipeline.raw_datasets["train"].to_dict())
    )
    validation_features = mocked_pipeline._prepare_validation_features(
        Batch(mocked_pipeline.raw_datasets["validation"].to_dict())
    )
    test_features = mocked_pipeline._prepare_validation_features(
        Batch(mocked_pipeline.raw_datasets["test"].to_dict())
    )

    # make relevant assertions regarding train features
    assert len(train_features["input_ids"]) == 12
    assert train_features["start_positions"][0] == 9
    assert train_features["end_positions"][0] == 10
    assert train_features["start_positions"][1] == 9
    assert train_features["end_positions"][1] == 9
    assert train_features["start_positions"][2] == 10
    assert train_features["end_positions"][2] == 11
    assert train_features["start_positions"][3] == 0
    assert train_features["end_positions"][3] == 0
    assert train_features["start_positions"][4] == 0
    assert train_features["end_positions"][4] == 0
    assert train_features["start_positions"][5] == 0
    assert train_features["end_positions"][5] == 0
    assert train_features["start_positions"][6] == 381
    assert train_features["end_positions"][6] == 386
    assert train_features["start_positions"][7] == 0
    assert train_features["end_positions"][7] == 0
    assert train_features["start_positions"][8] == 0
    assert train_features["end_positions"][8] == 0
    assert train_features["start_positions"][9] == 139
    assert train_features["end_positions"][9] == 144
    assert train_features["start_positions"][10] == 387
    assert train_features["end_positions"][10] == 392
    assert train_features["start_positions"][11] == 13
    assert train_features["end_positions"][11] == 18

    # make relevant assertions regarding validation features
    assert len(validation_features["input_ids"]) == 12
    assert validation_features["example_id"] == [
        "validation_id_1",
        "validation_id_2",
        "validation_id_3",
        "validation_id_4",
        "validation_id_5",
        "validation_id_5",
        "validation_id_6",
        "validation_id_6",
        "validation_id_7",
        "validation_id_7",
        "validation_id_8",
        "validation_id_8",
    ]
    validation_context_indices = [
        [index for index, mapping in enumerate(example) if mapping is not None]
        for example in validation_features["offset_mapping"]
    ]
    for index in range(0, 11, 2):
        if index == 0:
            assert min(validation_context_indices[index]) == 9
            assert max(validation_context_indices[index]) == 13
            assert min(validation_context_indices[index + 1]) == 9
            assert max(validation_context_indices[index + 1]) == 13
        elif index == 2:
            assert min(validation_context_indices[index]) == 9
            assert max(validation_context_indices[index]) == 14
            assert min(validation_context_indices[index + 1]) == 9
            assert max(validation_context_indices[index + 1]) == 14
        else:
            assert min(validation_context_indices[index]) == 9
            assert max(validation_context_indices[index]) == 510
            assert min(validation_context_indices[index + 1]) == 9
            assert max(validation_context_indices[index + 1]) == 294

    # make relevant assertions regarding test features
    assert len(test_features["input_ids"]) == 12
    assert test_features["example_id"] == [
        "test_id_1",
        "test_id_2",
        "test_id_3",
        "test_id_4",
        "test_id_5",
        "test_id_5",
        "test_id_6",
        "test_id_6",
        "test_id_7",
        "test_id_7",
        "test_id_8",
        "test_id_8",
    ]
    test_context_indices = [
        [index for index, mapping in enumerate(example) if mapping is not None]
        for example in test_features["offset_mapping"]
    ]
    for index in range(0, 11, 2):
        if index == 0:
            assert min(test_context_indices[index]) == 9
            assert max(test_context_indices[index]) == 13
            assert min(test_context_indices[index + 1]) == 9
            assert max(test_context_indices[index + 1]) == 13
        elif index == 2:
            assert min(test_context_indices[index]) == 9
            assert max(test_context_indices[index]) == 14
            assert min(test_context_indices[index + 1]) == 9
            assert max(test_context_indices[index + 1]) == 14
        else:
            assert min(test_context_indices[index]) == 9
            assert max(test_context_indices[index]) == 510
            assert min(test_context_indices[index + 1]) == 9
            assert max(test_context_indices[index + 1]) == 294


def test__post_processing_function(mocked_qa_examples, mocked_arguments, mocker):
    # create mocked pipeline object
    mocked_pipeline = Reading_Comprehension_Pipeline(*mocked_arguments())
    mocked_pipeline.answer_column_name = "answers"
    mocked_qa_examples_test_subset = mocked_qa_examples["test"].select(range(3))
    mocked_predictions = {
        "test_id_1": "answer",
        "test_id_2": "for",
        "test_id_3": "PolicyQA",
    }

    # create mocked objects
    postprocess_qa_predictions = mocker.patch(
        "reading_comprehension.Reading_Comprehension_Pipeline"
        "._postprocess_qa_predictions",
        return_value=(mocked_predictions, None),
    )

    # execute relevant pipeline method
    eval_prediction = mocked_pipeline._post_processing_function(
        mocked_qa_examples_test_subset,
        None,
        mocked_predictions,
    )

    # make relevant assertions
    postprocess_qa_predictions.assert_called_once_with(
        examples=mocked_qa_examples_test_subset,
        features=None,
        predictions=mocked_predictions,
    )
    assert eval_prediction.predictions == [
        {"id": "test_id_1", "prediction_text": "answer"},
        {"id": "test_id_2", "prediction_text": "for"},
        {"id": "test_id_3", "prediction_text": "PolicyQA"},
    ]
    assert eval_prediction.label_ids == [
        {
            "id": "test_id_1",
            "answers": {
                "answer_start": [0, 5],
                "text": ["test answer", "answer for PolicyQA"],
            },
        },
        {
            "id": "test_id_2",
            "answers": {"answer_start": [0, 5], "text": ["test", "answer"]},
        },
        {"id": "test_id_3", "answers": {"answer_start": [8], "text": ["test answer"]}},
    ]


@pytest.mark.parametrize(
    "wrong_prediction_len, wrong_prediction_shape",
    [(True, False), (False, True), (False, False)],
)
def test__postprocess_qa_predictions(
    wrong_prediction_len,
    wrong_prediction_shape,
    mocked_qa_examples_features_predictions,
    mocked_arguments,
    mocker,
):
    # create mocked pipeline object
    mocked_pipeline = Reading_Comprehension_Pipeline(*mocked_arguments())

    # parse examples, features and predictions
    (
        examples,
        features,
        predictions,
        _,
    ) = mocked_qa_examples_features_predictions

    # mock relevant methods
    mocked_pipeline_logger = mocker.patch.object(mocked_pipeline, "logger", create=True)

    # execute relevant pipeline method conditionally
    if wrong_prediction_len:
        with pytest.raises(ValueError):
            mocked_pipeline._postprocess_qa_predictions(
                examples, features, predictions[0]
            )
    elif wrong_prediction_shape:
        with pytest.raises(ValueError):
            mocked_pipeline._postprocess_qa_predictions(
                examples, features[:1], predictions
            )
    else:
        all_predictions, all_nbest_json = mocked_pipeline._postprocess_qa_predictions(
            examples, features, predictions
        )

        # make assertions
        mocked_pipeline_logger.info.assert_called_once()
        assert all_predictions == {
            "sample_id_1": "sample answer for PolicyQA",
            "sample_id_2": "for PolicyQA",
            "sample_id_3": "extra text",
        }
        assert len(all_nbest_json["sample_id_1"]) == 15
        assert len(all_nbest_json["sample_id_2"]) == 20
        assert len(all_nbest_json["sample_id_3"]) == 20


@pytest.mark.parametrize(
    "n_best_size",
    [1, 2],
)
def test__postprocess_qa_predictions_no_predictions(
    n_best_size,
    mocked_qa_examples_features_predictions,
    mocked_arguments,
    mocker,
):
    # create mocked pipeline object
    mocked_pipeline = Reading_Comprehension_Pipeline(
        *mocked_arguments(n_best_size=n_best_size)
    )

    # parse examples, features and predictions
    (
        examples,
        features,
        predictions,
        _,
    ) = mocked_qa_examples_features_predictions

    # mock relevant methods
    mocked_pipeline_logger = mocker.patch.object(mocked_pipeline, "logger", create=True)

    # execute relevant pipeline method
    all_predictions, all_nbest_json = mocked_pipeline._postprocess_qa_predictions(
        examples, features, (np.roll(predictions[0], 6), np.roll(predictions[1], 6))
    )

    # make assertions
    mocked_pipeline_logger.info.assert_called_once()
    assert all_predictions == {
        "sample_id_1": "empty",
        "sample_id_2": "empty",
        "sample_id_3": "empty",
    }
    assert len(all_nbest_json["sample_id_1"]) == 1
    assert len(all_nbest_json["sample_id_2"]) == 1
    assert len(all_nbest_json["sample_id_3"]) == 1


def test__set_metrics(mocked_arguments, mocker):
    # create mocked pipeline object
    mocked_pipeline = Reading_Comprehension_Pipeline(*mocked_arguments())

    # create mocked objects
    evaluate_load = mocker.patch(
        "reading_comprehension.evaluate.load", return_value="squad"
    )

    # execute relevant pipeline method
    mocked_pipeline._set_metrics()

    # make assertions
    evaluate_load.assert_called_once_with("squad")
    assert mocked_pipeline.metric == "squad"
    assert mocked_pipeline.train_args.metric_for_best_model == "f1"
    assert mocked_pipeline.train_args.greater_is_better


def test__compute_metrics(mocked_arguments, mocker):
    # create mocked pipeline object
    mocked_pipeline = Reading_Comprehension_Pipeline(*mocked_arguments())

    # create mocked objects
    metric = mocker.patch.object(mocked_pipeline, "metric", create=True)

    # execute relevant pipeline method
    mocked_pipeline._compute_metrics(
        mocker.MagicMock(predictions="sample_predictions", label_ids="sample_label_ids")
    )

    # make assertions
    metric.compute.assert_called_once_with(
        predictions="sample_predictions", references="sample_label_ids"
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
    [3, 5],
)
def test__run_train_loop(
    do_train,
    do_eval,
    do_predict,
    early_stopping_patience,
    mocked_qa_examples,
    mocked_arguments,
    mocker,
):
    # create mocked pipeline object
    mocked_pipeline = Reading_Comprehension_Pipeline(
        *mocked_arguments(
            do_train=do_train,
            do_eval=do_eval,
            do_predict=do_predict,
            early_stopping_patience=early_stopping_patience,
        )
    )
    mocked_pipeline.raw_datasets = mocked_qa_examples
    mocked_pipeline.model = "model"
    mocked_pipeline.train_dataset = [1, 2, 3]
    mocked_pipeline.eval_dataset = [4, 5, 6]
    mocked_pipeline.predict_dataset = [7, 8, 9]
    mocked_pipeline.tokenizer = "tokenizer"
    mocked_pipeline.data_collator = "data_collator"
    mocked_pipeline._post_processing_function = "_post_processing_function"
    mocked_pipeline._compute_metrics = "_compute_metrics"
    mocked_pipeline.last_checkpoint = "last_checkpoint"

    # create mocked objects
    qa_trainer = mocker.patch(
        "reading_comprehension.QuestionAnsweringTrainer",
    )
    qa_trainer.return_value.configure_mock(
        **{
            "train.return_value": SimpleNamespace(metrics={}),
            "evaluate.return_value": {},
            "predict.return_value": SimpleNamespace(
                metrics={}, predictions=[10, 11, 12]
            ),
        }
    )
    early_stopping_callback = mocker.patch(
        "reading_comprehension.EarlyStoppingCallback",
    )
    json_dump = mocker.patch("reading_comprehension.json.dump")
    mocker.patch("reading_comprehension.open")
    mocker.patch.object(mocked_pipeline, "logger", create=True)

    # execute relevant pipeline method
    mocked_pipeline._run_train_loop()

    # make common assertion
    qa_trainer.assert_called_once_with(
        model="model",
        args=mocker.ANY,
        train_dataset=[1, 2, 3] if do_train else None,
        eval_dataset=[4, 5, 6] if do_eval else None,
        eval_examples=mocked_qa_examples["validation"] if do_eval else None,
        tokenizer="tokenizer",
        data_collator="data_collator",
        post_process_function="_post_processing_function",
        compute_metrics="_compute_metrics",
        callbacks=[
            early_stopping_callback(early_stopping_patience=early_stopping_patience)
        ],
    )

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
            [7, 8, 9], mocked_qa_examples["test"]
        )
        mocked_pipeline.trainer.log_metrics.assert_any_call(
            "predict", {"predict_samples": 3}
        )
        mocked_pipeline.trainer.save_metrics.assert_any_call(
            "predict", {"predict_samples": 3}
        )
        json_dump.assert_called_once_with(
            [10, 11, 12],
            mocker.ANY,
        )
    else:
        mocked_pipeline.trainer.predict.assert_not_called()
        with pytest.raises(AssertionError):
            mocked_pipeline.trainer.log_metrics.assert_any_call("predict", mocker.ANY)
        with pytest.raises(AssertionError):
            mocked_pipeline.trainer.save_metrics.assert_any_call("predict", mocker.ANY)
        json_dump.assert_not_called()
