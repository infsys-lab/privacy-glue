#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import logging
import numpy as np
import evaluate
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    default_data_collator,
)

from parser import DataArguments, ModelArguments
from utils.pipeline_utils import Privacy_GLUE_Pipeline
from utils.model_utils import MultiTaskModel

logger = logging.getLogger(__name__)


class Sequence_Tagging_Pipeline(Privacy_GLUE_Pipeline):
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
        if self.data_args.task == "piextract":
            from tasks.piextract import SUBTASKS as subtasks
        elif self.data_args.task == "policy_ie_b":
            from tasks.policy_ie_b import SUBTASKS as subtasks
        else:
            logger.warn(
                f"Task: {self.data_args.task} is not an instance of \
                multitask learning problem!"
            )
            subtasks = [self.data_args.task]

        self.subtasks = subtasks

    def _retrieve_data(self) -> None:
        self.raw_datasets = self._get_data()
        if self.train_args.do_train:
            self.train_dataset = self.raw_datasets["train"]

        if self.train_args.do_eval:
            self.eval_dataset = self.raw_datasets["validation"]

        if self.train_args.do_predict:
            self.predict_dataset = self.raw_datasets["test"]

    def _load_pretrained_model_and_tokenizer(self) -> None:
        if self.train_args.do_train:
            self.label_names = self.train_dataset.features["tags"].feature.names

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.tokenizer_name
            if self.model_args.tokenizer_name
            else self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            use_fast=True,
            revision=self.model_args.model_revision,
        )

        self.model = MultiTaskModel(
            self.model_args.model_name_or_path, self.subtasks, self.label_names
        )

    def _apply_preprocessing(self) -> None:
        self.data_args.label_all_tokens = False
        # Map that sends B-Xxx label to its I-Xxx counterpart
        b_to_i_label = {}
        for st in self.subtasks:
            label_list = self.label_names[st]
            b_to_i_label[st] = []
            for idx, label in enumerate(label_list):
                if label.startswith("B-") and label.replace("B-", "I-") in label_list:
                    b_to_i_label[st].append(label_list.index(label.replace("B-", "I-")))
                else:
                    b_to_i_label[st].append(idx)
        label_to_ids = {
            l: i for st in self.subtasks for i, l in enumerate(self.label_names[st])
        }

        def preprocess_function(examples):
            # padding = self.model.encoder.config.max_length
            # Tokenize the texts
            tokenized_inputs = self.tokenizer(
                examples["tokens"],
                padding="max_length" if self.data_args.pad_to_max_length else False,
                max_length=self.data_args.max_seq_length,
                truncation=True,
                is_split_into_words=True,
            )
            # Warn if seqeuence length choice is not logical
            if self.data_args.max_seq_length > self.tokenizer.model_max_length:
                self.logger.warning(
                    f"The max_seq_length passed ({self.data_args.max_seq_length}) "
                    "is larger than the maximum length for the "
                    f"model ({self.tokenizer.model_max_length}). "
                    f"Using max_seq_length={self.tokenizer.model_max_length}"
                )
            self.max_seq_length = min(
                self.data_args.max_seq_length, self.tokenizer.model_max_length
            )
            # transform labels to label_ids
            labels = []
            task_ids = []
            for i, (st, label) in enumerate(zip(examples["subtask"], examples["tags"])):
                task_id = self.subtasks.index(st)
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    # Special tokens have a word id that is None.
                    # We set the label to -100 so they are automatically
                    # ignored in the loss function.
                    if word_idx is None:
                        label_ids.append(-100)
                    # We set the label for the first token of each word.
                    elif word_idx != previous_word_idx:
                        find_dot = label[word_idx].find(".")
                        if find_dot > 0:
                            w_label = label[word_idx][:find_dot]
                        else:
                            w_label = label[word_idx]

                        label_ids.append(label_to_ids[w_label])

                    # For the other tokens in a word, we set the label to either the
                    # current label or -100, depending on the label_all_tokens flag.
                    else:
                        if self.data_args.label_all_tokens:
                            find_dot = label[word_idx].find(".")
                            if find_dot > 0:
                                w_label = label[word_idx][:find_dot]
                            else:
                                w_label = label[word_idx]
                            label_ids.append(b_to_i_label[st][label_to_ids[w_label]])
                        else:
                            label_ids.append(-100)
                    previous_word_idx = word_idx
                labels.append(label_ids)
                task_ids.append(task_id)

            tokenized_inputs["labels"] = labels
            tokenized_inputs["task_ids"] = task_ids
            return tokenized_inputs

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
                    num_proc=self.data_args.preprocessing_num_workers,
                    desc="Running tokenizer on train dataset",
                    remove_columns=["tokens", "tags", "subtask"],
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
                    num_proc=self.data_args.preprocessing_num_workers,
                    desc="Running tokenizer on validation dataset",
                    remove_columns=["tokens", "tags", "subtask"],
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
                    num_proc=self.data_args.preprocessing_num_workers,
                    desc="Running tokenizer on prediction dataset",
                    remove_columns=["tokens", "tags", "subtask"],
                )

        # datacollator
        self.data_collator = (
            default_data_collator
            if self.data_args.pad_to_max_length
            else DataCollatorWithPadding(
                self.tokenizer,
                pad_to_multiple_of=8 if self.train_args.fp16 else None,
            )
        )

    def _set_metrics(self) -> None:
        # Get the metric function
        metric = evaluate.load("seqeval")

        def compute_f1(p: EvalPrediction):
            labels = p.label_ids
            predictions = (
                p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            )
            predictions = np.argmax(predictions, axis=2)
            # Remove ignored index (special tokens)
            true_predictions = {st: [] for st in self.subtasks}
            true_labels = {st: [] for st in self.subtasks}
            per_tasks_metrics = {}
            for i, (prediction, label) in enumerate(zip(predictions, labels)):
                true_label_vec = []
                true_pred_vec = []
                # get the subtasks according to mod of index
                task_name = self.subtasks[i % len(self.subtasks)]
                for (pr, la) in zip(prediction, label):
                    if la != -100:
                        true_label = self.label_names[task_name][la]
                        true_pred = self.label_names[task_name][pr]
                        true_pred_vec.append(true_pred)
                        true_label_vec.append(true_label)
                true_predictions[task_name].append(true_pred_vec)
                true_labels[task_name].append(true_label_vec)

            for st in self.subtasks:
                m = metric.compute(
                    predictions=true_predictions[st],
                    references=true_labels[st],
                )
                per_tasks_metrics[st] = m
            return {
                "precision": np.mean(
                    [per_tasks_metrics[st]["overall_precision"] for st in self.subtasks]
                ),
                "recall": np.mean(
                    [per_tasks_metrics[st]["overall_recall"] for st in self.subtasks]
                ),
                "f1": np.mean(
                    [per_tasks_metrics[st]["overall_f1"] for st in self.subtasks]
                ),
                "accuracy": np.mean(
                    [per_tasks_metrics[st]["overall_accuracy"] for st in self.subtasks]
                ),
            }

        self.compute_metrics = compute_f1

    def _run_train_loop(self) -> None:
        # Initialize the Trainer
        self.train_args.include_inputs_for_metrics = True
        self.trainer = Trainer(
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

            predictions = [
                self.label_names[item] for item in np.argmax(predictions, axis=2)
            ]

            output_predict_file = os.path.join(
                self.train_args.output_dir, "predictions.txt"
            )
            if self.trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    writer.write("index\tprediction\ttrue_label\n")
                    for index, (item, true_l) in enumerate(zip(predictions, labels)):
                        writer.write(f"{index}\t{item}\t{true_l}\n")
