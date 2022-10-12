#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from wandb.util import generate_id
from sequence_classification import Sequence_Classification_Pipeline
from sequence_tagging import Sequence_Tagging_Pipeline
from reading_comprehension import Reading_Comprehension_Pipeline
from parser import TASKS, get_parser
import os
import re


def summarize(model_dir: str) -> None:
    pass


def main() -> None:
    # get parser and parse arguments
    parser = get_parser()
    data_args, model_args, train_args = parser.parse_args_into_dataclasses()

    # capture base output directory
    output_dir = train_args.output_dir
    model_dir = os.path.join(
        output_dir,
        re.sub(r"[/-]", "_", model_args.model_name_or_path),
    )

    # decide iteration strategy
    if data_args.task != "all":
        tasks = [data_args.task]
    else:
        tasks = [task for task in TASKS if task != "all"]

    # loop over tasks and seeds
    for task in tasks:
        data_args.task = task
        model_args.wandb_group_id = (
            f"experiment_{generate_id()}" if "wandb" in train_args.report_to else None
        )
        for seed in range(model_args.random_seed_iterations):
            train_args.seed = seed
            train_args.output_dir = os.path.join(
                model_dir,
                re.sub(r"[/-]", "_", data_args.task),
                f"seed_{seed}",
            )
            # branch into separate workflows depending on task type
            if data_args.task in [
                "opp_115",
                "policy_detection",
                "policy_ie_a",
                "privacy_qa",
            ]:
                Sequence_Classification_Pipeline(
                    data_args, model_args, train_args
                ).run_pipeline()
            elif data_args.task in ["piextract", "policy_ie_b"]:
                Sequence_Tagging_Pipeline(
                    data_args, model_args, train_args
                ).run_pipeline()
            elif data_args.task == "policy_qa":  # pragma: no branch
                Reading_Comprehension_Pipeline(
                    data_args, model_args, train_args
                ).run_pipeline()

    # summarize PrivacyGLUE benchmark
    if model_args.do_summarize:
        summarize(model_dir)


if __name__ == "__main__":
    main()
