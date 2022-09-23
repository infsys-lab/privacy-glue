#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from parser import TASKS, get_parser
import os
import re


def summarize(model_dir: str) -> None:
    raise NotImplementedError


def main() -> None:
    # get parser and parse arguments
    parser = get_parser()
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()

    # configure results reporting
    use_wandb = False
    if train_args.report_to == "wandb":
        import wandb

        use_wandb = True
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
        if use_wandb:
            os.environ["WANDB_RUN_GROUP"] = f"experiment_{wandb.util.generate_id()}"
        for seed in range(model_args.random_seed_iterations):
            data_args.task = task
            train_args.seed = seed
            train_args.output_dir = os.path.join(
                model_dir,
                re.sub(r"[/-]", "_", data_args.task),
                "seed_%s" % seed,
            )
            if use_wandb:
                wandb_run = wandb.init(
                    name=f'{os.environ["WANDB_RUN_GROUP"][11:]}\
                        _seed_{str(seed)}',
                    project="privacyGLUE-" + task,
                    reinit=True,
                )
            # branch into separate workflows depending on task type
            if data_args.task in [
                "opp_115",
                "policy_detection",
                "policy_ie_a",
                "privacy_qa",
            ]:
                raise NotImplementedError
            elif data_args.task in ["piextract", "policy_ie_b"]:
                raise NotImplementedError
            elif data_args.task == "policy_qa":
                raise NotImplementedError
            if use_wandb:
                wandb_run.finish()

    # summarize PrivacyGLUE benchmark
    if model_args.do_summarize:
        summarize(model_dir)


if __name__ == "__main__":
    main()
