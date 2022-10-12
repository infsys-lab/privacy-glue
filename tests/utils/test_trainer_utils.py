#!/usr/bin/env python
# -*- coding: utf-8 -*-

from types import SimpleNamespace
from transformers import TrainingArguments
from utils.trainer_utils import QuestionAnsweringTrainer
import pytest


@pytest.fixture
def mocked_qa_trainer(mocked_regression_model, mocked_regression_config, tmp_path):
    return QuestionAnsweringTrainer(
        model=mocked_regression_model(mocked_regression_config()),
        args=TrainingArguments(output_dir=tmp_path),
    )


def test__init__(mocked_qa_trainer, mocked_arguments):
    assert mocked_qa_trainer.eval_examples is None
    assert mocked_qa_trainer.post_process_function is None


@pytest.mark.parametrize(
    "compute_metrics",
    [True, False],
)
@pytest.mark.parametrize(
    "post_process_function",
    [True, False],
)
def test_evaluate(compute_metrics, post_process_function, mocked_qa_trainer, mocker):
    # create relevant mocks
    mocked_qa_trainer_eval_loop = mocker.patch.object(
        mocked_qa_trainer,
        "evaluation_loop",
        return_value=SimpleNamespace(predictions=[4, 5, 6]),
    )
    mocked_qa_trainer_prediction_loop = mocker.patch.object(
        mocked_qa_trainer,
        "prediction_loop",
    )
    mocked_qa_trainer_log = mocker.patch.object(mocked_qa_trainer, "log", create=True)
    mocked_qa_callback_handler = mocker.patch.object(
        mocked_qa_trainer, "callback_handler", create=True
    )
    if post_process_function:
        mocked_qa_trainer_post_process_function = mocker.patch.object(
            mocked_qa_trainer,
            "post_process_function",
            create=True,
            return_value=[1, 2, 3],
        )
    else:
        mocked_qa_trainer.post_process_function = None
    if compute_metrics:
        mocked_qa_trainer_compute_metrics = mocker.patch.object(
            mocked_qa_trainer,
            "compute_metrics",
            create=True,
            return_value={
                "f1": [0.0, 0.0, 0.75],
                "exact_match": [0.0, 0.0, 0.0],
                "eval_sample_metric": ["test"],
            },
        )
    else:
        mocked_qa_trainer.compute_metrics = None

    # execute relevant trainer method
    metrics = mocked_qa_trainer.evaluate(
        eval_dataset=[1, 2, 3], eval_examples=[1, 2, 3]
    )

    # make common assertions
    mocked_qa_trainer_prediction_loop.assert_not_called()

    # make conditional assertions about compute metrics
    if compute_metrics:
        mocked_qa_trainer_eval_loop.assert_called_once_with(
            mocker.ANY,
            description=mocker.ANY,
            prediction_loss_only=None,
            ignore_keys=mocker.ANY,
        )
    else:
        mocked_qa_trainer_eval_loop.assert_called_once_with(
            mocker.ANY,
            description=mocker.ANY,
            prediction_loss_only=True,
            ignore_keys=mocker.ANY,
        )

    # make conditional assertions about metrics
    if compute_metrics and post_process_function:
        assert metrics == {
            "eval_f1": [0.0, 0.0, 0.75],
            "eval_exact_match": [0.0, 0.0, 0.0],
            "eval_sample_metric": ["test"],
        }
        mocked_qa_trainer_post_process_function.assert_called_once_with(
            [1, 2, 3], [1, 2, 3], [4, 5, 6]
        )
        mocked_qa_trainer_compute_metrics.assert_called_once_with([1, 2, 3])
        mocked_qa_trainer_log.assert_called_once_with(
            {
                "eval_f1": [0.0, 0.0, 0.75],
                "eval_exact_match": [0.0, 0.0, 0.0],
                "eval_sample_metric": ["test"],
            }
        )
        mocked_qa_callback_handler.on_evaluate.assert_called_once_with(
            mocker.ANY,
            mocker.ANY,
            mocker.ANY,
            {
                "eval_f1": [0.0, 0.0, 0.75],
                "eval_exact_match": [0.0, 0.0, 0.0],
                "eval_sample_metric": ["test"],
            },
        )
    else:
        assert metrics == {}
        mocked_qa_trainer_log.assert_not_called()
        mocked_qa_callback_handler.on_evaluate.assert_called_once_with(
            mocker.ANY,
            mocker.ANY,
            mocker.ANY,
            {},
        )


@pytest.mark.parametrize(
    "compute_metrics",
    [True, False],
)
@pytest.mark.parametrize(
    "post_process_function",
    [True, False],
)
def test_predict(compute_metrics, post_process_function, mocked_qa_trainer, mocker):
    # create relevant mocks
    mocked_qa_trainer_eval_loop = mocker.patch.object(
        mocked_qa_trainer,
        "evaluation_loop",
        return_value=SimpleNamespace(predictions=[4, 5, 6]),
    )
    mocked_qa_trainer_prediction_loop = mocker.patch.object(
        mocked_qa_trainer,
        "prediction_loop",
    )
    if post_process_function:
        mocked_qa_trainer_post_process_function = mocker.patch.object(
            mocked_qa_trainer,
            "post_process_function",
            create=True,
            return_value=SimpleNamespace(predictions=[1, 2, 3], label_ids=[1, 0, 0]),
        )
    else:
        mocked_qa_trainer.post_process_function = None
    if compute_metrics:
        mocked_qa_trainer_compute_metrics = mocker.patch.object(
            mocked_qa_trainer,
            "compute_metrics",
            create=True,
            return_value={
                "f1": [0.0, 0.0, 0.75],
                "exact_match": [0.0, 0.0, 0.0],
                "predict_sample_metric": ["test"],
            },
        )
    else:
        mocked_qa_trainer.compute_metrics = None

    # execute relevant trainer method
    output = mocked_qa_trainer.predict(
        predict_dataset=[1, 2, 3], predict_examples=[1, 2, 3]
    )

    # make common assertions
    mocked_qa_trainer_prediction_loop.assert_not_called()

    # make conditional assertions about compute metrics
    if compute_metrics:
        mocked_qa_trainer_eval_loop.assert_called_once_with(
            mocker.ANY,
            description=mocker.ANY,
            prediction_loss_only=None,
            ignore_keys=mocker.ANY,
        )
    else:
        mocked_qa_trainer_eval_loop.assert_called_once_with(
            mocker.ANY,
            description=mocker.ANY,
            prediction_loss_only=True,
            ignore_keys=mocker.ANY,
        )

    # make conditional assertions about metrics
    if compute_metrics and post_process_function:
        assert output.predictions == [1, 2, 3]
        assert output.label_ids == [1, 0, 0]
        assert output.metrics == {
            "predict_f1": [0.0, 0.0, 0.75],
            "predict_exact_match": [0.0, 0.0, 0.0],
            "predict_sample_metric": ["test"],
        }
        mocked_qa_trainer_post_process_function.assert_called_once_with(
            [1, 2, 3],
            [1, 2, 3],
            [4, 5, 6],
        )
        mocked_qa_trainer_compute_metrics.assert_called_once_with(
            SimpleNamespace(predictions=[1, 2, 3], label_ids=[1, 0, 0])
        )
    else:
        assert output.predictions == [4, 5, 6]
