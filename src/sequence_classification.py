#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import numpy as np
import evaluate
import torch
import logging

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    default_data_collator,
)


from parser import DataArguments, ModelArguments
from utils.trainer_utils import Weighted_Random_Sampler_Trainer
from utils.pipeline_utils import Privacy_GLUE_Pipeline

logger = logging.getLogger(__name__)

TASK2PROBLEM_TYPE = {"opp_115": "multi_label"}
TASK2INPUT_KEYS = {"privacy_qa": ["question", "text"]}


class Sequence_Classification_Pipeline(Privacy_GLUE_Pipeline):
    """
    Subclass of abstract Privacy_GLUE_Pipeline class with implementations
    of task specific pipeline functions.
    """

    def __init__(
        self,
        data_args: DataArguments,
        model_args: ModelArguments,
        training_args: TrainingArguments,
    ) -> None:
        super().__init__(data_args, model_args, training_args)
        self.problem_type = TASK2PROBLEM_TYPE.get(self.data_args.task, "single_label")
        self.input_keys = TASK2INPUT_KEYS.get(self.data_args.task, ["text"])

    def _retrieve_data(self) -> None:
        data = self._get_data()
        self.label_names = data["train"].features["label"].names

        if self.train_args.do_train:
            self.train_dataset = data["train"]

        if self.train_args.do_eval:
            self.eval_dataset = data["validation"]

        if self.train_args.do_predict:
            self.predict_dataset = data["test"]

    def _load_pretrained_model_and_tokenizer(self) -> None:
        # Load pretrained model and tokenizer
        # In distributed training, the .from_pretrained methods guarantee that only one
        # local process can concurrently download model & vocab.
        self.config = AutoConfig.from_pretrained(
            self.model_args.config_name
            if self.model_args.config_name
            else self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
            problem_type=self.problem_type + "_classification",
            num_labels=len(self.label_names),
            id2label=dict(enumerate(self.label_names)),
            label2id={l: n for n, l in enumerate(self.label_names)},
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.tokenizer_name
            if self.model_args.tokenizer_name
            else self.model_args.model_name_or_path,
            # do_lower_case=self.model_args.do_lower_case,
            cache_dir=self.model_args.cache_dir,
            use_fast=self.model_args.use_fast_tokenizer,
            revision=self.model_args.model_revision,
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_args.model_name_or_path,
            from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
            config=self.config,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
            # ignore_mismatched_sizes=self.model_args.ignore_mismatched_sizes,
        )

    def _apply_preprocessing(self) -> None:
        padding = False

        def preprocess_function(examples):
            args = tuple([examples[key] for key in self.input_keys])
            # Tokenize the texts
            return self.tokenizer(
                *args,
                padding=padding,
                truncation=True,
            )

        if self.train_args.do_train:
            if self.data_args.max_train_samples is not None:
                max_train_samples = min(
                    len(self.train_dataset), self.data_args.max_train_samples
                )
                self.train_dataset = self.train_dataset.select(range(max_train_samples))

            with self.train_args.main_process_first(
                desc="train dataset map pre-processing"
            ):
                self.train_dataset = self.train_dataset.map(
                    preprocess_function,
                    batched=True,
                    load_from_cache_file=not self.data_args.overwrite_cache,
                    desc="Running tokenizer on train dataset",
                    remove_columns=self.input_keys,
                )

        if self.train_args.do_eval:
            if self.data_args.max_eval_samples is not None:
                max_eval_samples = min(
                    len(self.eval_dataset), self.data_args.max_eval_samples
                )
                self.eval_dataset = self.eval_dataset.select(range(max_eval_samples))
            with self.train_args.main_process_first(
                desc="validation dataset map pre-processing"
            ):
                self.eval_dataset = self.eval_dataset.map(
                    preprocess_function,
                    batched=True,
                    load_from_cache_file=not self.data_args.overwrite_cache,
                    desc="Running tokenizer on validation dataset",
                    remove_columns=self.input_keys,
                )

        if self.train_args.do_predict:
            if self.data_args.max_predict_samples is not None:
                max_predict_samples = min(
                    len(self.predict_dataset), self.data_args.max_predict_samples
                )
                self.predict_dataset = self.predict_dataset.select(
                    range(max_predict_samples)
                )
            with self.train_args.main_process_first(
                desc="prediction dataset map pre-processing"
            ):
                self.predict_dataset = self.predict_dataset.map(
                    preprocess_function,
                    batched=True,
                    load_from_cache_file=not self.data_args.overwrite_cache,
                    desc="Running tokenizer on prediction dataset",
                    remove_columns=self.input_keys,
                )

        self.data_collator = (
            default_data_collator
            if self.data_args.pad_to_max_length
            else DataCollatorWithPadding(
                self.tokenizer,
                pad_to_multiple_of=8 if self.train_args.fp16 else None,
            )
        )

    def _set_metrics(self) -> None:
        # set training arguments for metrics
        self.train_args.metric_for_best_model = "f1"
        self.train_args.greater_is_better = True

        # Get the metric function
        if self.problem_type == "multi_label":
            metric = evaluate.load("f1", "multilabel")
            # using sample f1
            averaging = "samples"

            def transform_to_binary(pred):
                return np.round(torch.special.expit(torch.Tensor(pred)))

        else:
            metric = evaluate.load("f1", "multiclass")
            averaging = "macro"

            def transform_to_binary(pred):
                return np.argmax(pred, axis=1)

        def compute_f1(p: EvalPrediction):
            preds = (
                p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            )
            m = metric.compute(
                predictions=transform_to_binary(preds),
                references=p.label_ids.astype("int32"),
                average=averaging,
            )
            return m

        self.compute_metrics = compute_f1

    def _run_train_loop(self) -> None:
        # Initialize the Trainer
        if self.data_args.task == "privacy_qa":
            this_task_trainer = Weighted_Random_Sampler_Trainer
        else:
            this_task_trainer = Trainer

        self.trainer = this_task_trainer(
            model=self.model,
            args=self.train_args,
            train_dataset=self.train_dataset if self.train_args.do_train else None,
            eval_dataset=self.eval_dataset if self.train_args.do_eval else None,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.model_args.early_stopping_patience
                )
            ],
        )
        # Training
        if self.train_args.do_train:
            train_result = self.trainer.train(
                resume_from_checkpoint=self.last_checkpoint
            )
            metrics = train_result.metrics
            metrics["train_samples"] = len(self.train_dataset)

            self.trainer.save_model()  # Saves the tokenizer too for easy upload

            self.trainer.log_metrics("train", metrics)
            self.trainer.save_metrics("train", metrics)
            self.trainer.save_state()

        # Evaluation
        if self.train_args.do_eval:
            logger.info("*** Evaluate ***")
            self.eval_metrics = self.trainer.evaluate(eval_dataset=self.eval_dataset)

            self.eval_metrics["eval_samples"] = len(self.eval_dataset)

            self.trainer.log_metrics("eval", self.eval_metrics)
            self.trainer.save_metrics("eval", self.eval_metrics)

        # Prediction
        if self.train_args.do_predict:
            logger.info("*** Predict ***")
            predictions, labels, self.predict_metrics = self.trainer.predict(
                self.predict_dataset, metric_key_prefix="predict"
            )

            self.predict_metrics["predict_samples"] = len(self.predict_dataset)

            self.trainer.log_metrics("predict", self.predict_metrics)
            self.trainer.log(self.predict_metrics)
            self.trainer.save_metrics("predict", self.predict_metrics)
            if self.problem_type == "multi_label":
                predictions = np.round(torch.special.expit(torch.Tensor(predictions)))
                predictions = [
                    ",".join(
                        [
                            self.label_names[idx]
                            for idx, item in enumerate(labels)
                            if int(item) == 1
                        ]
                    )
                    for labels in predictions
                ]
            else:
                predictions = [
                    self.label_names[item] for item in np.argmax(predictions, axis=1)
                ]

            output_predict_file = os.path.join(
                self.train_args.output_dir, "predictions.txt"
            )
            if self.trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    writer.write("index\tprediction\ttrue_label\n")
                    for index, (item, true_l) in enumerate(zip(predictions, labels)):
                        writer.write(f"{index}\t{item}\t{true_l}\n")
