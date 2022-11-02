#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import subprocess
from glob import glob

import pytest
import torch
from transformers.trainer_utils import get_last_checkpoint

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
        pytest.param("cpu", ["python3"], ["--no_cuda"], marks=require_cpu),
        pytest.param(
            "gpu",
            ["python3"],
            [
                "--fp16",
                "--fp16_full_eval",
            ],
            marks=require_single_gpu,
        ),
        pytest.param(
            "multi_gpu",
            ["torchrun", "--nproc_per_node", str(torch.cuda.device_count())],
            [
                "--fp16",
                "--fp16_full_eval",
            ],
            marks=require_multi_gpu,
        ),
    ],
)
class Test_Reproducibility:
    def get_cli_arguments(self, task, model_name_or_path, output_dir):
        return {
            "--task": task,
            "--model_name_or_path": model_name_or_path,
            "--output_dir": output_dir,
            "--do_train": None,
            "--do_eval": None,
            "--do_pred": None,
            "--do_summarize": None,
            "--max_train_samples": "16",
            "--max_eval_samples": "16",
            "--max_predict_samples": "16",
            "--load_best_model_at_end": None,
            "--evaluation_strategy": "epoch",
            "--save_strategy": "epoch",
            "--save_total_limit": "2",
            "--num_train_epochs": "4",
            "--per_device_train_batch_size": "2",
            "--per_device_eval_batch_size": "2",
            "--random_seed_iterations": "1",
            "--report_to": "none",
            "--overwrite_cache": None,
            "--early_stopping_patience": str(int(1e5)),
        }

    def convert_dict_to_flat_list(self, input_dict):
        return [
            sub_item for item in input_dict.items() for sub_item in item if sub_item
        ]

    def test_full_train_reproducibility(
        self, task, model_name_or_path, device, runtime, extra_args, tmp_path
    ):
        # initialize CLI arguments
        args_list_one = (
            self.convert_dict_to_flat_list(
                self.get_cli_arguments(
                    task, model_name_or_path, os.path.join(str(tmp_path), "run_one")
                )
            )
            + extra_args
        )
        args_list_two = (
            self.convert_dict_to_flat_list(
                self.get_cli_arguments(
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
        subprocess.run(
            runtime + ["src/privacy_glue.py"] + args_list_one, env=subprocess_env
        )
        subprocess.run(
            runtime + ["src/privacy_glue.py"] + args_list_two, env=subprocess_env
        )

        # load dumped benchmark summary one
        with open(
            glob(os.path.join(str(tmp_path), "run_one", "*", "benchmark_summary.json"))[
                0
            ]
        ) as input_file_stream:
            benchmark_summary_one = json.load(input_file_stream)

        # load dumped benchmark summary two
        with open(
            glob(os.path.join(str(tmp_path), "run_two", "*", "benchmark_summary.json"))[
                0
            ]
        ) as input_file_stream:
            benchmark_summary_two = json.load(input_file_stream)

        # get latest checkpoint directories
        checkpoint_one = get_last_checkpoint(
            glob(
                os.path.join(str(tmp_path), "run_one", "**", "seed_0"), recursive=True
            )[0]
        )
        checkpoint_two = get_last_checkpoint(
            glob(
                os.path.join(str(tmp_path), "run_two", "**", "seed_0"), recursive=True
            )[0]
        )

        # check weights of final models
        weights_one = torch.load(os.path.join(checkpoint_one, "pytorch_model.bin"))
        weights_two = torch.load(os.path.join(checkpoint_two, "pytorch_model.bin"))

        # assert that benchmark summaries are the same
        assert benchmark_summary_one == benchmark_summary_two

        # assert that keys are the same
        assert weights_one.keys() == weights_two.keys()

        # now assert values of keys are the same
        for key in weights_one.keys():
            assert torch.allclose(weights_one[key], weights_two[key])
