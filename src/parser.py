#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from transformers.hf_argparser import DataClassType
from transformers import HfArgumentParser, TrainingArguments
from dataclasses import dataclass, field
from typing import Optional
import os

# 1. NOTE: Fields with no default value set will be transformed
# into`required arguments within the HuggingFace argument parser
# 2. NOTE: Enum-type objects will be transformed into choices


MODELS = [
    "sentence-transformers/all-mpnet-base-v2",
    "bert-base-uncased",
    "nlpaueb/legal-bert-base-uncased",
    "mukund/privbert",
]


TASKS = [
    "opp_115",
    "piextract",
    "policy_detection",
    "policy_ie_a",
    "policy_ie_b",
    "policy_qa",
    "privacy_qa",
    "all",
]


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from "
            "huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as "
            "model_name_or_path"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, "
            "tag name or commit id)."
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as "
            "model_name_or_path"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded "
            "from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the "
            "tokenizers library) or not"
        },
    )
    random_seed_iterations: int = field(
        default=5, metadata={"help": "Number of random seed iterations to run"}
    )
    early_stopping_patience: int = field(
        default=3, metadata={"help": "Early stopping patience value"}
    )
    do_summarize: bool = field(
        default=False, metadata={"help": "Summarize over all random seeds"}
    )
    do_clean: bool = field(
        default=False, metadata={"help": "Clean all old checkpoints after training"}
    )

    def __post_init__(self):
        assert self.model_name_or_path in MODELS, (
            f"Model '{self.model_name_or_path}' is not supported, "
            f"please select model from {MODELS}"
        )


@dataclass
class DataArguments:
    task: str = field(metadata={"help": "The name of the task for fine-tuning"})
    data_dir: str = field(
        default=os.path.relpath(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        ),
        metadata={"help": "Path to directory containing task input data"},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={
            "help": "Overwrite the cached training, evaluation and prediction sets"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the "
            "number of training examples to this value if set"
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the "
            "number of evaluation examples to this value if set"
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the "
            "number of prediction examples to this value if set"
        },
    )

    def __post_init__(self):
        assert os.path.isdir(self.data_dir), f"{self.data_dir} is not a valid directory"
        assert (
            self.task in TASKS
        ), f"Task '{self.task}' is not supported, please select task from {TASKS}"


def get_parser() -> HfArgumentParser:
    return HfArgumentParser(
        (
            DataClassType(ModelArguments),
            DataClassType(DataArguments),
            DataClassType(TrainingArguments),
        )
    )
