#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from datasets import DatasetDict
from transformers import TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint
from parser import DataArguments, ModelArguments
from tasks.opp_115 import load_opp_115
from tasks.piextract import load_piextract
from tasks.policy_detection import load_policy_detection
from tasks.policy_ie_a import load_policy_ie_a
from tasks.policy_ie_b import load_policy_ie_b
from tasks.policy_qa import load_policy_qa
from tasks.privacy_qa import load_privacy_qa
from utils.logging_utils import init_logger, add_file_handler, remove_all_file_handlers
import transformers
import datasets
import logging
import torch
import os


class SuccessFileFoundException(Exception):
    pass


class _Privacy_GLUE_Pipeline(ABC):
    def __init__(
        self,
        data_args: DataArguments,
        model_args: ModelArguments,
        train_args: TrainingArguments,
        success_file: str = ".success",
    ) -> None:
        self.data_args = data_args
        self.model_args = model_args
        self.train_args = train_args
        self.success_file = success_file

    def _get_data(self) -> DatasetDict:
        # define new argument based on task name
        task_dir = os.path.join(self.data_args.data_dir, self.data_args.task)

        # load dataset based on task name
        if self.data_args.task == "opp_115":
            data = load_opp_115(task_dir)
        elif self.data_args.task == "piextract":
            data = load_piextract(task_dir)
        elif self.data_args.task == "policy_detection":
            data = load_policy_detection(task_dir)
        elif self.data_args.task == "policy_ie_a":
            data = load_policy_ie_a(task_dir)
        elif self.data_args.task == "policy_ie_b":
            data = load_policy_ie_b(task_dir)
        elif self.data_args.task == "policy_qa":
            data = load_policy_qa(task_dir)
        elif self.data_args.task == "privacy_qa":
            data = load_privacy_qa(task_dir)

        return data

    def create_run_dir(self) -> None:
        # create output_dir if it does not exit
        os.makedirs(self.train_args.output_dir, exist_ok=True)

    def init_root_logger(self) -> None:
        # initialize root logger
        self.logger = logging.getLogger()
        init_logger(self.logger, self.train_args.get_process_log_level())
        add_file_handler(
            self.logger,
            self.train_args.get_process_log_level(),
            os.path.join(self.train_args.output_dir, "session.log"),
        )

    def init_third_party_loggers(self) -> None:
        # configure datasets logger
        # NOTE: if working with distributed training, logging/saving would only
        # need to be configured on the main or zero process
        datasets.utils.logging.set_verbosity(self.train_args.get_process_log_level())
        remove_all_file_handlers(datasets.utils.logging.get_logger())
        add_file_handler(
            datasets.utils.logging.get_logger(),
            self.train_args.get_process_log_level(),
            os.path.join(self.train_args.output_dir, "session.log"),
        )

        # configure transformers logger
        # NOTE: if working with distributed training, logging/saving would only
        # need to be configured on the main or zero process
        transformers.utils.logging.set_verbosity(
            self.train_args.get_process_log_level()
        )
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
        remove_all_file_handlers(transformers.utils.logging.get_logger())
        add_file_handler(
            transformers.utils.logging.get_logger(),
            self.train_args.get_process_log_level(),
            os.path.join(self.train_args.output_dir, "session.log"),
        )

    def check_for_success_file(self) -> None:
        # check for existing exit code and decide action
        if (
            os.path.exists(os.path.join(self.train_args.output_dir, self.success_file))
            and not self.train_args.overwrite_output_dir
            and self.train_args.do_train
        ):
            message = (
                "%s file found; therefore training already complete" % self.success_file
            )
            self.logger.info(message)
            raise SuccessFileFoundException(message)

    def dump_misc_args(self) -> None:
        # dump miscellaneous arguments
        torch.save(
            {"model_args": self.model_args, "data_args": self.data_args},
            os.path.join(self.train_args.output_dir, "misc_args.bin"),
        )

    def log_starting_arguments(self) -> None:
        # log on each process the small summary
        self.logger.info(
            (
                "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, "
                "16-bits training: %s"
            )
            % (
                self.train_args.local_rank,
                self.train_args.device,
                self.train_args.n_gpu,
                bool(self.train_args.local_rank != -1),
                self.train_args.fp16,
            )
        )

    def set_global_seeds(self) -> None:
        # set seed before initializing model
        set_seed(self.train_args.seed)

    def find_existing_checkpoint(self) -> None:
        # detect last checkpoint if necessary
        if self.train_args.do_train and not self.train_args.overwrite_output_dir:
            # use upstream function for detection
            self.checkpoint = get_last_checkpoint(self.train_args.output_dir)

            # check if checkpoint exists
            if self.checkpoint is not None:
                self.logger.warning(
                    "Checkpoint detected, resuming training from %s. "
                    "To avoid this behavior, change --output_dir or "
                    "add --overwrite_output_dir to train from scratch" % self.checkpoint
                )
        else:
            self.checkpoint = None

    def save_success_file(self) -> None:
        with open(
            os.path.join(self.train_args.output_dir, self.success_file), "w"
        ) as output_file_stream:
            output_file_stream.write("%s\n" % 0)

    def clean_logger(self) -> None:
        self.logger.handlers = []

    @abstractmethod
    def retrieve_data(self) -> None:
        pass

    @abstractmethod
    def load_pretrained_model_and_tokenizer(self) -> None:
        pass

    @abstractmethod
    def apply_preprocessing(self) -> None:
        pass

    @abstractmethod
    def set_metrics(self) -> None:
        pass

    @abstractmethod
    def run_train_loop(self) -> None:
        pass

    @abstractmethod
    def destroy(self) -> None:
        pass

    def run_start(self) -> None:
        self.create_run_dir()
        self.init_root_logger()
        self.init_third_party_loggers()
        self.check_for_success_file()
        self.dump_misc_args()
        self.log_starting_arguments()
        self.set_global_seeds()
        self.find_existing_checkpoint()

    def run_task(self) -> None:
        self.retrieve_data()
        self.load_pretrained_model_and_tokenizer()
        self.apply_preprocessing()
        self.set_metrics()
        self.run_train_loop()
        self.destroy()

    def run_end(self) -> None:
        self.save_success_file()

    def run_finally(self) -> None:
        self.clean_logger()

    def run_pipeline(self) -> None:
        try:
            self.run_start()
            self.run_task()
            self.run_end()
        except SuccessFileFoundException:
            pass
        finally:
            self.run_finally()
