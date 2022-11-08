#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import shutil
from abc import ABC, abstractmethod
from functools import wraps
from glob import glob
from parser import DataArguments, ModelArguments

import datasets
import torch
import transformers
from datasets import DatasetDict
from transformers import TrainingArguments
from transformers.trainer_utils import enable_full_determinism, get_last_checkpoint

import wandb
from tasks.opp_115 import load_opp_115
from tasks.piextract import load_piextract
from tasks.policy_detection import load_policy_detection
from tasks.policy_ie_a import load_policy_ie_a
from tasks.policy_ie_b import load_policy_ie_b
from tasks.policy_qa import load_policy_qa
from tasks.privacy_qa import load_privacy_qa
from utils.logging_utils import add_file_handler, init_logger


def main_process_first_only(function):
    @wraps(function)
    def wrapper(self, *args, **kwargs):
        with self.train_args.main_process_first():
            if self.train_args.local_rank in [-1, 0]:
                return function(self, *args, **kwargs)

    return wrapper


class SuccessFileFoundException(Exception):
    pass


class Privacy_GLUE_Pipeline(ABC):
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
        elif self.data_args.task == "privacy_qa":  # pragma: no branch
            data = load_privacy_qa(task_dir)

        return data

    @main_process_first_only
    def _init_run_dir(self) -> None:
        if (
            os.path.exists(self.train_args.output_dir)
            and self.train_args.overwrite_output_dir
        ):
            # delete run directory if it exists
            shutil.rmtree(self.train_args.output_dir)

        # create output_dir if it does not exit
        os.makedirs(self.train_args.output_dir, exist_ok=True)

    def _init_root_logger(self) -> None:
        # initialize root logger
        self.logger = logging.getLogger()
        init_logger(self.logger, self.train_args.get_process_log_level())
        add_file_handler(
            self.logger,
            self.train_args.get_process_log_level(),
            os.path.join(self.train_args.output_dir, "session.log"),
        )

    def _init_third_party_loggers(self) -> None:
        # set logger verbosity
        datasets.utils.logging.set_verbosity(self.train_args.get_process_log_level())
        transformers.utils.logging.set_verbosity(
            self.train_args.get_process_log_level()
        )

        # disable any default handlers since we take the root logger's
        transformers.utils.logging.disable_default_handler()

        # allow for propagation to the root logger to prevent double configurations
        datasets.utils.logging.enable_propagation()
        transformers.utils.logging.enable_propagation()

    def _check_for_success_file(self) -> None:
        # check for existing exit code and decide action
        if (
            os.path.exists(os.path.join(self.train_args.output_dir, self.success_file))
            and self.train_args.do_train
        ):
            message = (
                f"{self.success_file} file found; therefore training already complete"
            )
            self.logger.info(message)
            raise SuccessFileFoundException(message)

    @main_process_first_only
    def _dump_misc_args(self) -> None:
        # dump miscellaneous arguments
        torch.save(
            {"data_args": self.data_args, "model_args": self.model_args},
            os.path.join(self.train_args.output_dir, "misc_args.bin"),
        )

    def _log_starting_arguments(self) -> None:
        # log summary and arguments in each process
        self.logger.info(
            (
                f"Process rank: {self.train_args.local_rank}, "
                f"device: {self.train_args.device}, "
                f"n_gpu: {self.train_args.n_gpu}, "
                f"distributed training: {bool(self.train_args.local_rank != -1)}, "
                f"16-bits training: {self.train_args.fp16}"
            )
        )
        self.logger.info(f"Data arguments: {self.data_args}")
        self.logger.info(f"Model arguments: {self.model_args}")
        self.logger.info(f"Training arguments: {self.train_args}")

    def _make_deterministic(self) -> None:
        # enable full determinism by setting seeds and other arguments
        enable_full_determinism(self.train_args.seed)

    def _find_existing_checkpoint(self) -> None:
        # detect last checkpoint if necessary
        if self.train_args.do_train:
            # use upstream function for detection
            self.last_checkpoint = get_last_checkpoint(self.train_args.output_dir)

            # check if checkpoint exists
            if self.last_checkpoint is not None:
                self.logger.warning(
                    "Checkpoint detected, resuming training from "
                    f"{self.last_checkpoint}. To avoid this behavior, change "
                    "--output_dir or add --overwrite_output_dir to train from scratch"
                )
        else:
            self.last_checkpoint = None

    @main_process_first_only
    def _init_wandb_run(self) -> None:
        if "wandb" in self.train_args.report_to:
            wandb.init(
                name=(
                    f"{self.model_args.wandb_group_id[11:]}"
                    f"_seed_{str(self.train_args.seed)}"
                ),
                group=self.model_args.wandb_group_id,
                project=f"privacyGLUE-{self.data_args.task}",
                reinit=True,
                resume=True if self.last_checkpoint else None,
            )

    @main_process_first_only
    def _clean_checkpoint_dirs(self) -> None:
        if self.model_args.do_clean and self.train_args.do_train:
            for checkpoint in glob(
                os.path.join(self.train_args.output_dir, "checkpoint*")
            ):
                shutil.rmtree(checkpoint)

    @main_process_first_only
    def _save_success_file(self) -> None:
        if self.train_args.do_train:
            with open(
                os.path.join(self.train_args.output_dir, self.success_file), "w"
            ) as output_file_stream:
                output_file_stream.write("0\n")

    def _clean_loggers(self) -> None:
        datasets.utils.logging.get_logger().handlers = []
        transformers.utils.logging.get_logger().handlers = []
        if hasattr(self, "logger"):
            self.logger.handlers = []

    @main_process_first_only
    def _close_wandb(self) -> None:
        if "wandb" in self.train_args.report_to and wandb.run is not None:
            wandb.run.finish()

    def _destroy(self) -> None:
        # some variables are not freed automatically by pytorch and can quickly
        # fill up memory.
        self.trainer = None
        del self

    @abstractmethod
    def _retrieve_data(self) -> None:
        pass

    @abstractmethod
    def _load_pretrained_model_and_tokenizer(self) -> None:
        pass

    @abstractmethod
    def _apply_preprocessing(self) -> None:
        pass

    @abstractmethod
    def _set_metrics(self) -> None:
        pass

    @abstractmethod
    def _run_train_loop(self) -> None:
        pass

    def run_start(self) -> None:
        self._init_run_dir()
        self._init_root_logger()
        self._init_third_party_loggers()
        self._check_for_success_file()
        self._dump_misc_args()
        self._log_starting_arguments()
        self._make_deterministic()
        self._find_existing_checkpoint()
        self._init_wandb_run()

    def run_task(self) -> None:
        self._retrieve_data()
        self._load_pretrained_model_and_tokenizer()
        self._apply_preprocessing()
        self._set_metrics()
        self._run_train_loop()

    def run_end(self) -> None:
        self._clean_checkpoint_dirs()
        self._save_success_file()

    def run_finally(self) -> None:
        self._clean_loggers()
        self._close_wandb()
        self._destroy()

    def run_pipeline(self) -> None:
        try:
            self.run_start()
            self.run_task()
            self.run_end()
        except SuccessFileFoundException:
            pass
        finally:
            self.run_finally()
