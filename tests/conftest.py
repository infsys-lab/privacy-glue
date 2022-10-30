#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import random
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock

import datasets
import numpy as np
import pytest
import torch
from transformers import PretrainedConfig, PreTrainedModel


def pytest_configure():
    # globally disable caching with datasets
    datasets.disable_caching()


def get_mocked_arguments(
    task="all",
    data_dir="/tmp/data",
    max_seq_length=512,
    max_train_samples=None,
    max_eval_samples=None,
    max_predict_samples=None,
    n_best_size=20,
    max_answer_length=30,
    preprocessing_num_workers=1,
    overwrite_cache=False,
    pad_to_max_length=True,
    doc_stride=128,
    model_name_or_path="bert-base-uncased",
    do_clean=True,
    config_name=None,
    tokenizer_name=None,
    cache_dir=None,
    model_revision="main",
    wandb_group_id="experiment_test",
    early_stopping_patience=5,
    do_train=True,
    do_eval=True,
    do_predict=True,
    output_dir="/tmp/runs",
    overwrite_output_dir=False,
    log_level="info",
    report_to=[],
    local_rank=-1,
    device="cpu",
    n_gpu=0,
    fp16=False,
    seed=0,
    no_cuda=True,
    use_legacy_prediction_loop=False,
    random_seed_iterations=5,
    do_summarize=True,
    with_experiment_args=False,
):
    data_args = SimpleNamespace(
        task=task,
        data_dir=data_dir,
        max_seq_length=max_seq_length,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
        max_predict_samples=max_predict_samples,
        n_best_size=n_best_size,
        max_answer_length=max_answer_length,
        preprocessing_num_workers=preprocessing_num_workers,
        overwrite_cache=overwrite_cache,
        pad_to_max_length=pad_to_max_length,
        doc_stride=doc_stride,
    )
    model_args = SimpleNamespace(
        model_name_or_path=model_name_or_path,
        do_clean=do_clean,
        config_name=config_name,
        tokenizer_name=tokenizer_name,
        cache_dir=cache_dir,
        model_revision=model_revision,
        wandb_group_id=wandb_group_id,
        early_stopping_patience=early_stopping_patience,
    )
    train_args = SimpleNamespace(
        do_train=do_train,
        do_eval=do_eval,
        do_predict=do_predict,
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        log_level=logging.getLevelName(log_level.upper()),
        get_process_log_level=lambda: logging.getLevelName(log_level.upper()),
        report_to=report_to,
        local_rank=local_rank,
        device=device,
        n_gpu=n_gpu,
        fp16=fp16,
        seed=seed,
        main_process_first=lambda *args, **kwargs: nullcontext(),
        is_world_process_zero=lambda *args, **kwargs: nullcontext(),
        no_cuda=no_cuda,
        use_legacy_prediction_loop=use_legacy_prediction_loop,
    )
    experiment_args = SimpleNamespace(
        random_seed_iterations=random_seed_iterations, do_summarize=do_summarize
    )
    all_args = (
        (data_args, model_args, train_args)
        if not with_experiment_args
        else (data_args, model_args, train_args, experiment_args)
    )
    return all_args


@pytest.fixture
def deep_mocker():
    class DeepMagicMock(MagicMock):
        def __call__(self, /, *args, **kwargs):
            args = deepcopy(args)
            kwargs = deepcopy(kwargs)
            return super().__call__(*args, **kwargs)

    return DeepMagicMock


@pytest.fixture
def mocked_arguments():
    return get_mocked_arguments


@pytest.fixture
def mocked_arguments_with_tmp_path(tmp_path):
    run_dir = os.path.join(str(tmp_path), "runs")
    data_dir = os.path.join(str(tmp_path), "data")
    os.makedirs(run_dir)
    os.makedirs(data_dir)
    return partial(
        get_mocked_arguments,
        output_dir=run_dir,
        data_dir=data_dir,
    )


class MockConfig(PretrainedConfig):
    def __init__(self, a=0, b=0, double_output=False, random_torch=True, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b
        self.double_output = double_output
        self.random_torch = random_torch
        self.hidden_size = 1


class MockPreTrainedModel(PreTrainedModel):
    config_class = MockConfig
    base_model_prefix = "regression"

    def __init__(self, config):
        super().__init__(config)
        self.a = torch.nn.Parameter(torch.tensor(config.a).float())
        self.b = torch.nn.Parameter(torch.tensor(config.b).float())
        self.random_torch = config.random_torch

    def forward(self, input_x, labels=None, text=None, **kwargs):
        y = input_x * self.a + self.b
        if self.random_torch:
            torch_rand = torch.randn(1).squeeze()
        np_rand = np.random.rand()
        rand_rand = random.random()

        if self.random_torch:
            y += 0.05 * torch_rand
        y += 0.05 * torch.tensor(np_rand + rand_rand)

        if labels is None:
            return (y,)
        loss = torch.nn.functional.mse_loss(y, labels)
        return (loss, y)


@pytest.fixture
def mocked_regression_config():
    return MockConfig


@pytest.fixture
def mocked_regression_model():
    return MockPreTrainedModel
