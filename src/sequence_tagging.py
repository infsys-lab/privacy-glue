#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from typing import List, Dict

import logging
import torch
import torch.nn as nn
import numpy as np
import evaluate
import datasets
from transformers import (
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)

from parser import DataArguments, ModelArguments
from utils.pipeline_utils import Privacy_GLUE_Pipeline

logger = logging.getLogger(__name__)

TASK2SUBTASKS = {
    "piextract": {
        "tasks": ["COLLECT", "NOT_COLLECT", "SHARE", "NOT_SHARE"],
        "labels": [
            ["SHARE"],
            ["COLLECT"],
            ["NOT_COLLECT"],
            ["NOT_SHARE"],
        ],
    },
    "policy_ie_b": {
        "tasks": ["type-I", "type-II"],
        "labels": [
            [
                "data-protector",
                "data-protected",
                "data-collector",
                "data-collected",
                "data-receiver",
                "data-retained",
                "data-holder",
                "data-provider",
                "data-sharer",
                "data-shared",
                "storage-place",
                "retention-period",
                "protect-against",
                "action",
            ],
            [
                "purpose-argument",
                "polarity",
                "method",
                "condition-argument",
            ],
        ],
    },
}


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

        # information about label names
        self.subtasks = TASK2SUBTASKS[data_args.task]["tasks"]
        self.general_labels = TASK2SUBTASKS[data_args.task]["labels"]
        self.label_names = {
            task: ["O"] + [f"{pre}-{label}" for pre in ["B", "I"] for label in labels]
            for task, labels in zip(self.subtasks, self.general_labels)
        }

    def _retrieve_data(self) -> None:
        data = self._get_data()
        if self.train_args.do_train:
            self.train_dataset = self._form_multitask_dataset(data["train"])

        if self.train_args.do_eval:
            self.eval_dataset = self._form_multitask_dataset(data["validation"])

        if self.train_args.do_predict:
            self.predict_dataset = self._form_multitask_dataset(data["test"])

    def _load_pretrained_model_and_tokenizer(self) -> None:
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

    def _form_multitask_dataset(self, ds):
        # only one label per example, split the data into multiple tasks
        multi_trainset = {"tokens": [], "ner_tags": [], "subtask": []}
        for i, st in enumerate(self.subtasks):
            for example in ds:
                multi_trainset["tokens"].append(example["tokens"])
                multi_trainset["ner_tags"].append(
                    [tag[i] for tag in example["ner_tags"]]
                )
                multi_trainset["subtask"].append(st)

        multi_trainset = datasets.Dataset.from_dict(multi_trainset)
        multi_trainset.shuffle(42)
        return multi_trainset

    def _apply_preprocessing(self) -> None:
        self.data_args.label_all_tokens = True
        # Padding strategy
        padding = "max_length"
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
                padding=padding,
                truncation=True,
                is_split_into_words=True,
            )
            labels = []
            task_ids = []
            for i, (st, label) in enumerate(
                zip(examples["subtask"], examples["ner_tags"])
            ):
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
                        label_ids.append(label_to_ids[label[word_idx]])
                    # For the other tokens in a word, we set the label to either the
                    # current label or -100, depending on the label_all_tokens flag.
                    else:
                        if self.data_args.label_all_tokens:
                            label_ids.append(
                                b_to_i_label[st][label_to_ids[label[word_idx]]]
                            )
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
                    remove_columns=["tokens", "ner_tags", "subtask"],
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
                    remove_columns=["tokens", "ner_tags", "subtask"],
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
                    remove_columns=["tokens", "ner_tags", "subtask"],
                )
        # Data collator will default to DataCollatorWithPadding, so we change it if we
        # already did the padding.
        # if self.data_args.pad_to_max_length:
        #    data_collator = default_data_collator
        if self.train_args.fp16:
            self.data_collator = DataCollatorWithPadding(
                self.tokenizer, pad_to_multiple_of=8
            )
        else:
            self.data_collator = None

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
            checkpoint = None
            if self.train_args.resume_from_checkpoint is not None:
                checkpoint = self.train_args.resume_from_checkpoint
            # elif last_checkpoint is not None:
            #    checkpoint = last_checkpoint
            train_result = self.trainer.train(resume_from_checkpoint=checkpoint)
            metrics = train_result.metrics
            max_train_samples = (
                self.data_args.max_train_samples
                if self.data_args.max_train_samples is not None
                else len(self.train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(self.train_dataset))

            self.trainer.save_model()  # Saves the tokenizer too for easy upload

            self.trainer.log_metrics("train", metrics)
            self.trainer.save_metrics("train", metrics)
            self.trainer.save_state()

        # Evaluation
        if self.train_args.do_eval:
            logger.info("*** Evaluate ***")
            self.eval_metrics = self.trainer.evaluate(eval_dataset=self.eval_dataset)

            max_eval_samples = (
                self.data_args.max_eval_samples
                if self.data_args.max_eval_samples is not None
                else len(self.eval_dataset)
            )
            self.eval_metrics["eval_samples"] = min(
                max_eval_samples, len(self.eval_dataset)
            )

            self.trainer.log_metrics("eval", self.eval_metrics)
            self.trainer.save_metrics("eval", self.eval_metrics)

        # Prediction
        if self.train_args.do_predict:
            logger.info("*** Predict ***")
            predictions, labels, self.predict_metrics = self.trainer.predict(
                self.predict_dataset, metric_key_prefix="predict"
            )

            max_predict_samples = (
                self.data_args.max_predict_samples
                if self.data_args.max_predict_samples is not None
                else len(self.predict_dataset)
            )
            self.predict_metrics["predict_samples"] = min(
                max_predict_samples, len(self.predict_dataset)
            )

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


# adapted from
# https://towardsdatascience.com/
# how-to-create-and-train-a-multi-task-transformer-model-18c54a146240
class MultiTaskModel(nn.Module):
    def __init__(self, encoder_name_or_path, tasks: List, label_names: Dict):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(encoder_name_or_path)

        self.output_heads = nn.ModuleDict()
        # one model several heads
        for task_id, task in enumerate(tasks):
            decoder = TokenClassificationHead(
                self.encoder.config.hidden_size, len(label_names[task])
            )
            # ModuleDict requires keys to be strings
            self.output_heads[str(task_id)] = decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        task_ids=None,
        **kwargs,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output, pooled_output = outputs[:2]

        unique_task_ids_list = torch.unique(task_ids).tolist()

        loss_list = []
        logits_list = []
        for unique_task_id in unique_task_ids_list:
            task_id_filter = task_ids == unique_task_id
            logits, task_loss = self.output_heads[str(unique_task_id)].forward(
                sequence_output[task_id_filter],
                pooled_output[task_id_filter],
                labels=None if labels is None else labels[task_id_filter],
                attention_mask=attention_mask[task_id_filter],
            )

            if labels is not None:
                loss_list.append(task_loss)

            logits_list.append(logits[0])
        # logits are only used for eval. and in case of eval the batch is not multi task
        # For training only the loss is used
        outputs = (torch.stack(logits_list), outputs[2:])

        if len(loss_list) > 0:
            loss = torch.stack(loss_list)
            outputs = (loss.mean(),) + outputs
        return outputs


class TokenClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_p=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

        self._init_weights()

    def _init_weights(self):
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()

    def forward(
        self, sequence_output, pooled_output, labels=None, attention_mask=None, **kwargs
    ):
        sequence_output_dropout = self.dropout(sequence_output)
        logits = self.classifier(sequence_output_dropout)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()

            labels = labels.long()

            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels),
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return logits, loss
