#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import warnings
from glob import glob

import numpy as np

from parser import TASKS
from reading_comprehension import Reading_Comprehension_Pipeline
from sequence_classification import Sequence_Classification_Pipeline
from sequence_tagging import Sequence_Tagging_Pipeline
from utils.pipeline_utils import main_process_first_only


class Privacy_GLUE_Experiment_Manager:
    task_metrics = {
        "opp_115": ["macro_f1", "micro_f1"],
        "piextract": [
            "macro_f1",
            "micro_f1",
            "COLLECT_macro_f1",
            "COLLECT_micro_f1",
            "NOT_COLLECT_macro_f1",
            "NOT_COLLECT_micro_f1",
            "SHARE_macro_f1",
            "SHARE_micro_f1",
            "NOT_SHARE_macro_f1",
            "NOT_SHARE_micro_f1",
        ],
        "policy_detection": ["macro_f1", "micro_f1"],
        "policy_ie_a": ["macro_f1", "micro_f1"],
        "policy_ie_b": [
            "macro_f1",
            "micro_f1",
            "type-I_macro_f1",
            "type-I_micro_f1",
            "type-II_macro_f1",
            "type-II_micro_f1",
        ],
        "policy_qa": ["sample_f1", "exact_match"],
        "privacy_qa": ["macro_f1", "micro_f1"],
    }

    def __init__(self, data_args, model_args, train_args, experiment_args):
        self.data_args = data_args
        self.model_args = model_args
        self.train_args = train_args
        self.experiment_args = experiment_args
        self.experiment_args.model_dir = os.path.join(
            self.train_args.output_dir,
            re.sub(r"[/-]", "_", self.model_args.model_name_or_path),
        )

    @main_process_first_only
    def summarize(self) -> None:
        # create dictionary used for collecting metrics
        benchmark_summary = {}

        # loop over all task directories available
        for task_dir in glob(os.path.join(self.experiment_args.model_dir, "*/")):
            # create list to collect metrics by seeds and then group
            metric_by_seed_group = []
            task = os.path.basename(os.path.normpath(task_dir))

            # if directory is not part of our tasks, ignore it
            if task not in self.task_metrics:
                continue
            else:
                # loop over all seed directories inside valid task directory
                for seed_dir in glob(os.path.join(task_dir, "seed_*")):
                    all_results_file = os.path.join(seed_dir, "all_results.json")

                    # load JSON results file to dictionary
                    if os.path.exists(all_results_file):
                        with open(all_results_file) as input_file_stream:
                            all_results = json.load(input_file_stream)

                        # add metrics to upper list
                        metric_by_seed_group.append(
                            [
                                all_results[f"predict_{metric}"]
                                for metric in self.task_metrics[task]
                            ]
                        )
                    else:
                        warnings.warn(
                            f"Results file {all_results_file} not found, "
                            "skipping directory"
                        )

            # convert seed-group order to group-seed
            metric_by_group_seed = list(zip(*metric_by_seed_group))
            benchmark_summary[task] = {"metrics": self.task_metrics[task]}
            benchmark_summary[task]["mean"] = [
                np.round(np.mean(metric_group), 8).item()
                for metric_group in metric_by_group_seed
            ]
            benchmark_summary[task]["std"] = [
                np.round(np.std(metric_group), 8).item()
                for metric_group in metric_by_group_seed
            ]
            benchmark_summary[task]["num_samples"] = len(metric_by_seed_group)

        # dump benchmark dictionary
        with open(
            os.path.join(self.experiment_args.model_dir, "benchmark_summary.json"), "w"
        ) as output_file_stream:
            json.dump(benchmark_summary, output_file_stream)

    def run_experiments(self) -> None:
        # decide iteration strategy
        if self.data_args.task != "all":
            tasks = [self.data_args.task]
        else:
            tasks = [task for task in TASKS if task != "all"]

        # create wandb_group class variable
        self.model_args.wandb_group = (
            self.model_args.model_name_or_path
            if "wandb" in self.train_args.report_to
            else None
        )

        # loop over tasks and seeds
        for task in tasks:
            self.data_args.task = task
            for seed in range(self.experiment_args.random_seed_iterations):
                self.train_args.seed = seed
                self.train_args.output_dir = os.path.join(
                    self.experiment_args.model_dir,
                    self.data_args.task,
                    f"seed_{seed}",
                )

                # branch into separate workflows depending on task type
                if self.data_args.task in [
                    "opp_115",
                    "policy_detection",
                    "policy_ie_a",
                    "privacy_qa",
                ]:
                    Sequence_Classification_Pipeline(
                        self.data_args, self.model_args, self.train_args
                    ).run_pipeline()
                elif self.data_args.task in ["piextract", "policy_ie_b"]:
                    Sequence_Tagging_Pipeline(
                        self.data_args, self.model_args, self.train_args
                    ).run_pipeline()
                elif self.data_args.task == "policy_qa":  # pragma: no branch
                    Reading_Comprehension_Pipeline(
                        self.data_args, self.model_args, self.train_args
                    ).run_pipeline()

        # summarize PrivacyGLUE benchmark
        if self.experiment_args.do_summarize:
            self.summarize()
