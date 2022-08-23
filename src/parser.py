#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

# 1. NOTE: Fields with no default value set will be transformed
# into`required arguments within the HuggingFace argument parser
# 2. NOTE: Enum-type objects will be transformed into choices


class Model(Enum):
    all_mpnet_base_v2: str = "all-mpnet-base-v2"
    bert_base_uncased: str = "bert-base-uncased"
    nlpaueb_legal_bert_base_uncased: str = "nlpaueb/legal-bert-base-uncased"
    mukund_privbert: str = "mukund/privbert"


class Task(Enum):
    app_350: str = "app_350"
    opp_115: str = "opp_115"
    piextract: str = "piextract"
    policy_detection: str = "policy_detection"
    policy_ie: str = "policy_ie"
    policy_qa: str = "policy_qa"
    privacy_qa: str = "privacy_qa"
    all: str = "all"


@dataclass
class ModelArguments:
    model_name_or_path: Model = field(
        metadata={
            "help":
            "Path to pretrained model or model identifier from "
            "huggingface.co/models"
        })
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Pretrained config name or path if not the same as "
            "model_name_or_path"
        })
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Pretrained tokenizer name or path if not the same as "
            "model_name_or_path"
        })
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Where do you want to store the pretrained models downloaded "
            "from huggingface.co"
        })
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help":
            "Whether to use one of the fast tokenizer (backed by the "
            "tokenizers library) or not"
        })
    random_seed_iterations: int = field(
        default=5,
        metadata={"help": "Number of random seed iterations to run"})
    early_stopping_patience: int = field(
        default=3, metadata={"help": "Early stopping patience value"})
    do_summarize: bool = field(
        default=False, metadata={"help": "Summarize over all random seeds"})
    do_clean: bool = field(
        default=False,
        metadata={"help": "Clean all old checkpoints after training"})


@dataclass
class DataArguments:
    task: Task = field(
        metadata={"help": "The name of the task for fine-tuning"})
    overwrite_cache: bool = field(
        default=False,
        metadata={
            "help":
            "Overwrite the cached training, evaluation and prediction sets"
        })
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help":
            "For debugging purposes or quicker training, truncate the "
            "number of training examples to this value if set"
        })
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help":
            "For debugging purposes or quicker training, truncate the "
            "number of evaluation examples to this value if set"
        })
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help":
            "For debugging purposes or quicker training, truncate the "
            "number of prediction examples to this value if set"
        })
