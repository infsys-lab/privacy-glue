#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from types import SimpleNamespace
from functools import partial
from unittest.mock import MagicMock
from copy import deepcopy
import datasets
import logging
import pytest
import os


def pytest_configure():
    # globally disable caching with datasets
    datasets.disable_caching()


def get_mocked_arguments(
    task="all",
    data_dir="/tmp/data",
    model_name_or_path="bert-base-uncased",
    do_train=True,
    do_summarize=True,
    random_seed_iterations=5,
    wandb_group_id="test",
    output_dir="/tmp/runs",
    overwrite_output_dir=False,
    log_level="info",
    report_to=[],
    local_rank=-1,
    device="cpu",
    n_gpu=0,
    fp16=False,
    seed=0,
):
    return (
        SimpleNamespace(task=task, data_dir=data_dir),
        SimpleNamespace(
            model_name_or_path=model_name_or_path,
            do_summarize=do_summarize,
            random_seed_iterations=random_seed_iterations,
            wandb_group_id=wandb_group_id,
        ),
        SimpleNamespace(
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
        ),
    )


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
