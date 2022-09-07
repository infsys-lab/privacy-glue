#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from transformers import TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint
from utils.logging_utils import (
    init_logger,
    add_file_handler,
    remove_all_file_handlers,
)
from parser import TASKS, ModelArguments, DataArguments, get_parser
import transformers
import datasets
import logging
import torch
import os
import re

# define global logger
LOGGER = logging.getLogger(__name__)

# define exit code file
SUCCESS_FILE = ".success"


def save_success_file(directory: str, code: int = 0) -> None:
    with open(os.path.join(directory, SUCCESS_FILE), "w") as output_file_stream:
        output_file_stream.write("%s\n" % code)


def summarize(model_dir: str) -> None:
    raise NotImplementedError


def train(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
) -> None:
    try:
        # create output_dir if it does not exit
        os.makedirs(training_args.output_dir, exist_ok=True)

        # configure local logger
        global LOGGER
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
            os.path.exists(os.path.join(training_args.output_dir, SUCCESS_FILE))
            and not training_args.overwrite_output_dir
            and training_args.do_train
        ):
            LOGGER.info(".success file found; therefore training already complete")
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

        # branch into separate workflows depending on task type
        if data_args.task in [
            "opp_115",
            "policy_detection",
            "policy_ie_a",
            "privacy_qa",
        ]:
            raise NotImplementedError
        elif data_args.task in ["pi_extract", "policy_ie_b"]:
            raise NotImplementedError
        elif data_args.task == "policy_qa":
            raise NotImplementedError

        # save success file if everything proceeds well
        save_success_file(training_args.output_dir)

    # cleanup logger regardless of execution workflow (except SIGKILL)
    finally:
        remove_all_file_handlers(LOGGER)


def main() -> None:
    # get parser and parse arguments
    parser = get_parser()
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # initialize global logger
    global LOGGER
    LOGGER = logging.getLogger()
    init_logger(LOGGER, training_args.get_process_log_level())

    # capture base output directory
    output_dir = training_args.output_dir
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
        for seed in range(model_args.random_seed_iterations):
            data_args.task = task
            training_args.seed = seed
            training_args.output_dir = os.path.join(
                model_dir,
                re.sub(r"[/-]", "_", data_args.task),
                "seed_%s" % seed,
            )
            train(model_args, data_args, training_args)

    # summarize PrivacyGLUE benchmark
    if model_args.do_summarize:
        summarize(model_dir)


if __name__ == "__main__":
    main()
