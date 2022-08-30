#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from transformers import TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint
from utils.logging_utils import (
    add_stream_handler,
    add_file_handler,
    remove_all_file_handlers,
)
from parser import ModelArguments, DataArguments, Task, get_parser
from tasks.opp_115 import load_opp_115
from tasks.piextract import load_piextract
from tasks.policy_detection import load_policy_detection
from tasks.policy_ie_a import load_policy_ie_a
from tasks.policy_ie_b import load_policy_ie_b
from tasks.policy_qa import load_policy_qa
from tasks.privacy_qa import load_privacy_qa
import transformers
import datasets
import logging
import torch
import os
import re

# define global logger
LOGGER = logging.getLogger(__name__)

# define all tasks
TASKS = [task for task in Task._member_names_ if task != "all"]

# define exit code file
EXIT_CODE_FILE = "exit_code"


def save_exit_code(filename: str, code: int = 0) -> None:
    with open(filename, "w") as output_file_stream:
        output_file_stream.write("%s\n" % code)


def summarize(model_dir: str) -> None:
    raise NotImplementedError


def train(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
) -> None:
    # create output_dir if it does not exit
    os.makedirs(training_args.output_dir, exist_ok=True)

    # configure local logger
    global LOGGER
    remove_all_file_handlers(LOGGER)
    add_file_handler(
        LOGGER,
        training_args.get_process_log_level(),
        os.path.join(training_args.output_dir, "session.log"),
    )

    # configure datasets logger
    # NOTE: if working with distributed training, logging/saving would only
    # need to be configured on the main or zero process
    datasets.utils.logging.set_verbosity(training_args.get_process_log_level())
    remove_all_file_handlers(datasets.utils.logging.get_logger())
    add_file_handler(
        datasets.utils.logging.get_logger(),
        training_args.get_process_log_level(),
        os.path.join(training_args.output_dir, "session.log"),
    )

    # configure transformers logger
    # NOTE: if working with distributed training, logging/saving would only
    # need to be configured on the main or zero process
    transformers.utils.logging.set_verbosity(training_args.get_process_log_level())
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    remove_all_file_handlers(transformers.utils.logging.get_logger())
    add_file_handler(
        transformers.utils.logging.get_logger(),
        training_args.get_process_log_level(),
        os.path.join(training_args.output_dir, "session.log"),
    )

    # check for existing exit code and decide action
    if (
        os.path.exists(os.path.join(training_args.output_dir, EXIT_CODE_FILE))
        and not training_args.overwrite_output_dir
        and training_args.do_train
    ):
        LOGGER.info("Exit-code 0: training already complete, exiting")
        return None

    # dump miscellaneous arguments
    torch.save(
        {"model_args": model_args, "data_args": data_args},
        os.path.join(training_args.output_dir, "misc_args.bin"),
    )

    # log on each process the small summary
    LOGGER.info(
        (
            "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, "
            "16-bits training: %s"
        )
        % (
            training_args.local_rank,
            training_args.device,
            training_args.n_gpu,
            bool(training_args.local_rank != -1),
            training_args.fp16,
        )
    )

    # set seed before initializing model
    set_seed(training_args.seed)

    # detect last checkpoint if necessary
    if training_args.do_train and not training_args.overwrite_output_dir:
        # use upstream function for detection
        checkpoint = get_last_checkpoint(training_args.output_dir)

        # check if checkpoint exists
        if checkpoint is not None:
            LOGGER.warning(
                "Checkpoint detected, resuming training from %s. "
                "To avoid this behavior, change --output_dir or "
                "add --overwrite_output_dir to train from scratch" % checkpoint
            )
    else:
        checkpoint = None

    # define new argument based on task name
    data_args.task_dir = os.path.join(data_args.data_dir, data_args.task)

    # load dataset based on task name
    if data_args.task == "opp_115":
        data = load_opp_115(data_args.task_dir)
    elif data_args.task == "piextract":
        data = load_piextract(data_args.task_dir)
    elif data_args.task == "policy_detection":
        data = load_policy_detection(data_args.task_dir)
    elif data_args.task == "policy_ie_a":
        data = load_policy_ie_a(data_args.task_dir)
    elif data_args.task == "policy_ie_b":
        data = load_policy_ie_b(data_args.task_dir)
    elif data_args.task == "policy_qa":
        data = load_policy_qa(data_args.task_dir)
    elif data_args.task == "privacy_qa":
        data = load_privacy_qa(data_args.task_dir)

    print(data)

    # NOTE: temporarily raise error
    raise NotImplementedError


def main(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
) -> None:
    # capture base output directory
    output_dir = training_args.output_dir
    model_dir = os.path.join(
        output_dir, re.sub(r"[/-]", "_", model_args.model_name_or_path)
    )

    # decide iteration strategy
    if data_args.task != "all":
        tasks = [data_args.task]
    else:
        tasks = TASKS

    # loop over tasks and seeds
    for task in tasks:
        for seed in range(model_args.random_seed_iterations):
            data_args.task = task
            training_args.seed = seed
            training_args.output_dir = os.path.join(
                model_dir, re.sub(r"[/-]", "_", data_args.task), "seed_%s" % seed
            )
            train(model_args, data_args, training_args)

    # summarize PrivacyGLUE benchmark
    if model_args.do_summarize:
        summarize(model_dir)


if __name__ == "__main__":
    parser = get_parser()
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    LOGGER = logging.getLogger()
    add_stream_handler(LOGGER, training_args.get_process_log_level())
    main(model_args, data_args, training_args)
