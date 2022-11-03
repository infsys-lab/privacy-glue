#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser, TrainingArguments
from transformers.hf_argparser import DataClassType

# 1. NOTE: Fields with no default value set will be transformed
# into`required arguments within the HuggingFace argument parser
# 2. NOTE: Enum-type objects will be transformed into choices


MODELS = [
    "bert-base-uncased",
    "roberta-base",
    "nlpaueb/legal-bert-base-uncased",
    "saibo/legal-roberta-base",
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
class ExperimentArguments:
    random_seed_iterations: int = field(
        default=5, metadata={"help": "Number of random seed iterations to run"}
    )
    do_summarize: bool = field(
        default=False, metadata={"help": "Summarize over all random seeds"}
    )


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
    early_stopping_patience: int = field(
        default=5, metadata={"help": "Early stopping patience value"}
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
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. "
                "Sequences longer than this will be truncated, sequences "
                "shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when "
                "batching to the maximum length in the batch "
                "(which can be faster on GPU but will be slower on TPU)."
            )
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, "
            "how much stride to take between chunks."
        },
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
    n_best_size: int = field(
        default=20,
        metadata={
            "help": "PolicyQA: The total number of n-best predictions to generate when "
            "looking for an answer."
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": (
                "PolicyQA: The maximum length of an answer that can be generated. "
                "This is needed because the start and end predictions "
                "are not conditioned on one another."
            )
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
            DataClassType(DataArguments),
            DataClassType(ModelArguments),
            DataClassType(TrainingArguments),
            DataClassType(ExperimentArguments),
        )
    )
