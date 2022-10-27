#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
from copy import deepcopy
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock

import datasets
import pytest


def pytest_configure():
    # globally disable caching with datasets
    datasets.disable_caching()


def get_mocked_arguments(
    task="all",
    data_dir="/tmp/data",
    model_name_or_path="bert-base-uncased",
    do_train=True,
    do_clean=True,
    do_summarize=True,
    random_seed_iterations=5,
    wandb_group_id="experiment_test",
    output_dir="/tmp/runs",
    overwrite_output_dir=False,
    log_level="info",
    report_to=[],
    local_rank=-1,
    device="cpu",
    n_gpu=0,
    fp16=False,
    seed=0,
    with_experiment_args=False,
):
    data_args = SimpleNamespace(task=task, data_dir=data_dir)
    model_args = SimpleNamespace(
        model_name_or_path=model_name_or_path,
        do_clean=do_clean,
        wandb_group_id=wandb_group_id,
    )
    train_args = SimpleNamespace(
        do_train=do_train,
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
