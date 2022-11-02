#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import subprocess
from glob import glob

import pytest
import torch

require_cpu = pytest.mark.skipif(
    torch.cuda.device_count() != 0,
    reason="GPU device(s) detected",
)

require_single_gpu = pytest.mark.skipif(
    torch.cuda.device_count() != 1 or "CUDA_VISIBLE_DEVICES" not in os.environ,
    reason="Single GPU not available or 'CUDA_VISIBLE_DEVICES' not declared",
)

require_multi_gpu = pytest.mark.skipif(
    torch.cuda.device_count() < 2 or "CUDA_VISIBLE_DEVICES" not in os.environ,
    reason="Multiple GPUs not available or 'CUDA_VISIBLE_DEVICES' not declared",
)


def get_cli_arguments(task, model_name_or_path, output_dir):
    return {
        "--task": task,
        "--model_name_or_path": model_name_or_path,
        "--output_dir": output_dir,
        "--do_train": None,
        "--do_eval": None,
        "--do_pred": None,
        "--max_train_samples": "16",
        "--max_eval_samples": "16",
        "--max_predict_samples": "16",
        "--do_summarize": None,
        "--load_best_model_at_end": None,
        "--evaluation_strategy": "epoch",
        "--save_strategy": "epoch",
        "--save_total_limit": "2",
        "--num_train_epochs": "4",
        "--per_device_train_batch_size": "2",
        "--per_device_eval_batch_size": "2",
        "--random_seed_iterations": "1",
        "--report_to": "none",
        "--overwrite_output_dir": None,
        "--overwrite_cache": None,
    }


def convert_dict_to_flat_list(input_dict):
    return [sub_item for item in input_dict.items() for sub_item in item if sub_item]


@pytest.mark.slow
@pytest.mark.parametrize(
    "task",
    [
        "opp_115",
        "policy_detection",
        "policy_ie_a",
        "privacy_qa",
        "piextract",
        "policy_ie_b",
        "policy_qa",
    ],
)
@pytest.mark.parametrize(
    "model_name_or_path",
    [
        "sentence-transformers/all-mpnet-base-v2",
        "bert-base-uncased",
        "nlpaueb/legal-bert-base-uncased",
        "mukund/privbert",
    ],
)
@pytest.mark.parametrize(
    "device, runtime, extra_args",
    [
        pytest.param("cpu", "python3", ["--no_cuda"], marks=require_cpu),
        pytest.param(
            "gpu",
            "python3",
            [
                "--fp16",
                "--fp16_full_eval",
            ],
            marks=require_single_gpu,
        ),
        pytest.param(
            "multi_gpu",
            "torchrun",
            [
                "--fp16",
                "--fp16_full_eval",
            ],
            marks=require_multi_gpu,
        ),
    ],
)
def test_reproducibility_across_seeds(
    task, model_name_or_path, device, runtime, extra_args, tmp_path
):
    # initialize CLI arguments
    args_list_one = (
        convert_dict_to_flat_list(
            get_cli_arguments(
                task, model_name_or_path, os.path.join(str(tmp_path), "run_one")
            )
        )
        + extra_args
    )
    args_list_two = (
        convert_dict_to_flat_list(
            get_cli_arguments(
                task, model_name_or_path, os.path.join(str(tmp_path), "run_two")
            )
        )
        + extra_args
    )

    # initialize subprocess env by copying basic environment
    subprocess_env = os.environ.copy()

    # conditionally update subprocess environment
    if device == "cpu":
        subprocess_env["CUDA_VISIBLE_DEVICES"] = ""

    # run experiments
    process_one = subprocess.run(
        [runtime, "src/privacy_glue.py"] + args_list_one, env=subprocess_env
    )
    process_two = subprocess.run(
        [runtime, "src/privacy_glue.py"] + args_list_two, env=subprocess_env
    )

    # load dumped benchmark summary one
    with open(
        glob(os.path.join(str(tmp_path), "run_one", "*", "benchmark_summary.json"))[0]
    ) as input_file_stream:
        benchmark_summary_one = json.load(input_file_stream)

    # load dumped benchmark summary two
    with open(
        glob(os.path.join(str(tmp_path), "run_two", "*", "benchmark_summary.json"))[0]
    ) as input_file_stream:
        benchmark_summary_two = json.load(input_file_stream)

    # make initial assertions
    assert process_one.returncode == 0
    assert process_two.returncode == 0
    assert benchmark_summary_one == benchmark_summary_two

    # check weights of final models
    weights_one = torch.load(
        glob(
            os.path.join(
                str(tmp_path),
                "run_one",
                "**",
                "seed_0",
                "checkpoint-32",
                "pytorch_model.bin",
            ),
            recursive=True,
        )[0]
    )
    weights_two = torch.load(
        glob(
            os.path.join(
                str(tmp_path),
                "run_two",
                "**",
                "seed_0",
                "checkpoint-32",
                "pytorch_model.bin",
            ),
            recursive=True,
        )[0]
    )

    # assert that keys are the same
    assert weights_one.keys() == weights_two.keys()

    # now assert values of keys are the same
    for key in weights_one.keys():
        assert torch.allclose(weights_one[key], weights_two[key])
