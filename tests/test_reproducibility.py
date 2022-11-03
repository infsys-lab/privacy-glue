#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import shutil
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
        "bert-base-uncased",
        "roberta-base",
        "nlpaueb/legal-bert-base-uncased",
        "saibo/legal-roberta-base",
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
            ["--fp16"],
            marks=require_single_gpu,
        ),
        pytest.param(
            "multi_gpu",
            ["torchrun", "--nproc_per_node", str(torch.cuda.device_count())],
            ["--fp16"],
            marks=require_multi_gpu,
        ),
    ],
)
class Test_Reproducibility:
    def get_cli_arguments(
        self, task, model_name_or_path, output_dir, save_total_limit=2
    ):
        args = {
            "--task": task,
            "--model_name_or_path": model_name_or_path,
            "--output_dir": output_dir,
            "--do_train": None,
            "--do_eval": None,
            "--do_pred": None,
            "--do_summarize": None,
            "--load_best_model_at_end": None,
            "--evaluation_strategy": "epoch",
            "--save_strategy": "epoch",
            "--num_train_epochs": "4",
            "--learning_rate": "3e-5",
            "--warmup_ratio": "0.1",
            "--report_to": "none",
            "--per_device_train_batch_size": "2",
            "--per_device_eval_batch_size": "2",
            "--max_train_samples": "16",
            "--max_eval_samples": "16",
            "--max_predict_samples": "16",
            "--overwrite_cache": None,
            "--random_seed_iterations": "1",
        }

        # add extra arguments conditionally
        if save_total_limit is not None:
            args["--save_total_limit"] = str(save_total_limit)

        return args

    def convert_dict_to_flat_list(self, input_dict):
        return [
            sub_item for item in input_dict.items() for sub_item in item if sub_item
        ]

    def get_checkpoints_descending_recency(self, output_dir):
        _re_checkpoint = re.compile(r"^checkpoint\-(\d+)$")
        checkpoints = [
            path
            for path in os.listdir(output_dir)
            if _re_checkpoint.search(path) is not None
            and os.path.isdir(os.path.join(output_dir, path))
        ]
        return (
            [
                os.path.join(output_dir, checkpoint)
                for checkpoint in sorted(
                    checkpoints,
                    key=lambda x: int(_re_checkpoint.search(x).groups()[0]),
                    reverse=True,
                )
            ]
            if checkpoints
            else None
        )

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
        process_one = subprocess.run(
            runtime + ["src/privacy_glue.py"] + args_list_one, env=subprocess_env
        )
        process_two = subprocess.run(
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

        # assert that both processes exited zero
        assert process_one.returncode == 0
        assert process_two.returncode == 0

        # assert that benchmark summaries are the same
        assert benchmark_summary_one == benchmark_summary_two

        # assert that keys are the same
        assert weights_one.keys() == weights_two.keys()

        # now assert values of keys are the same
        for key in weights_one.keys():
            assert torch.allclose(weights_one[key], weights_two[key])

    def test_resume_train_reproducibility(
        self, task, model_name_or_path, device, runtime, extra_args, tmp_path
    ):
        # initialize CLI arguments
        args_list_one = (
            self.convert_dict_to_flat_list(
                self.get_cli_arguments(
                    task,
                    model_name_or_path,
                    os.path.join(str(tmp_path), "run_one"),
                    save_total_limit=None,
                )
            )
            + extra_args
        )
        args_list_two = (
            self.convert_dict_to_flat_list(
                self.get_cli_arguments(
                    task,
                    model_name_or_path,
                    os.path.join(str(tmp_path), "run_two"),
                    save_total_limit=None,
                )
            )
            + extra_args
        )

        # initialize subprocess env by copying basic environment
        subprocess_env = os.environ.copy()

        # conditionally update subprocess environment
        if device == "cpu":
            subprocess_env["CUDA_VISIBLE_DEVICES"] = ""

        # run full training experiment
        process_one = subprocess.run(
            runtime + ["src/privacy_glue.py"] + args_list_one, env=subprocess_env
        )

        # run partial training experiment
        process_two = subprocess.run(
            runtime + ["src/privacy_glue.py"] + args_list_two, env=subprocess_env
        )

        # remove .success file to simulate re-training
        os.remove(
            glob(
                os.path.join(str(tmp_path), "run_two", "**", "seed_0", ".success"),
                recursive=True,
            )[0]
        )

        # get produced checkpoints from latest to earliest
        checkpoints_two = self.get_checkpoints_descending_recency(
            glob(
                os.path.join(str(tmp_path), "run_two", "**", "seed_0"),
                recursive=True,
            )[0]
        )

        # skip this test if there are insufficient checkpoints
        if len(checkpoints_two) < 2 or checkpoints_two is None:
            pytest.skip(
                "Insufficient checkpoints to test reproducibility with resume training"
            )

        # delete recent half of checkpoints
        for checkpoint in checkpoints_two[: len(checkpoints_two) // 2]:
            shutil.rmtree(checkpoint)

        # run resumed training experiment
        process_two_prime = subprocess.run(
            runtime + ["src/privacy_glue.py"] + args_list_two, env=subprocess_env
        )

        # read second training log
        with open(
            glob(
                os.path.join(str(tmp_path), "run_two", "**", "session.log"),
                recursive=True,
            )[0]
        ) as input_file_stream:
            session_log_two = input_file_stream.read()

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

        # assert that all processes exited zero
        assert process_one.returncode == 0
        assert process_two.returncode == 0
        assert process_two_prime.returncode == 0

        # assert that training was indeed resumed
        assert "Checkpoint detected, resuming training" in session_log_two

        # assert that benchmark summaries are the same
        assert benchmark_summary_one == benchmark_summary_two

        # assert that keys are the same
        assert weights_one.keys() == weights_two.keys()

        # now assert values of keys are the same
        for key in weights_one.keys():
            assert torch.allclose(weights_one[key], weights_two[key])
