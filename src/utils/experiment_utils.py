#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from parser import TASKS

from wandb.util import generate_id

from reading_comprehension import Reading_Comprehension_Pipeline
from sequence_classification import Sequence_Classification_Pipeline
from sequence_tagging import Sequence_Tagging_Pipeline


class Privacy_GLUE_Experiment_Manager:
    def __init__(self, data_args, model_args, train_args, experiment_args):
        self.data_args = data_args
        self.model_args = model_args
        self.train_args = train_args
        self.experiment_args = experiment_args

    def _summarize(self) -> None:
        pass

    def run_experiments(self) -> None:
        # capture base output directory
        output_dir = self.train_args.output_dir
        self.experiment_args.model_dir = os.path.join(
            output_dir,
            re.sub(r"[/-]", "_", self.model_args.model_name_or_path),
        )

        # decide iteration strategy
        if self.data_args.task != "all":
            tasks = [self.data_args.task]
        else:
            tasks = [task for task in TASKS if task != "all"]

        # loop over tasks and seeds
        for task in tasks:
            self.data_args.task = task
            self.model_args.wandb_group_id = (
                f"experiment_{generate_id()}"
                if "wandb" in self.train_args.report_to
                else None
            )
            for seed in range(self.experiment_args.random_seed_iterations):
                self.train_args.seed = seed
                self.train_args.output_dir = os.path.join(
                    self.experiment_args.model_dir,
                    re.sub(r"[/-]", "_", self.data_args.task),
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
            self._summarize()
