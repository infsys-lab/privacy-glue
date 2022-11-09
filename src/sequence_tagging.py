#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from typing import Dict, List, Tuple

import numpy as np
from seqeval.metrics import sequence_labeling as seqeval_metrics
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from parser import DataArguments, ModelArguments
from utils.model_utils import MultiTaskModel
from utils.pipeline_utils import Privacy_GLUE_Pipeline
from utils.task_utils import sorted_interleave_task_datasets


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
            subtasks = [self.data_args.task]

        self.subtasks = subtasks

    def _retrieve_data(self) -> None:
        self.raw_datasets = self._get_data()
        self.label_names = {}
        for st in self.subtasks:
            self.label_names[st] = (
                self.raw_datasets["train"][st].features["tags"].feature.names
            )

        for split in ["train", "validation", "test"]:
            self.raw_datasets[split] = sorted_interleave_task_datasets(
                self.raw_datasets[split], delete_features=True
            )

    def _load_pretrained_model_and_tokenizer(self) -> None:
        self.config = AutoConfig.from_pretrained(
            self.model_args.config_name
            if self.model_args.config_name
            else self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
        )
        # We need to instantiate RobertaTokenizerFast with add_prefix_space=True
        # to use it with pretokenized inputs.
        add_prefix_space = self.config.__class__.__name__.startswith("Roberta")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.tokenizer_name
            if self.model_args.tokenizer_name
            else self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            use_fast=True,
            revision=self.model_args.model_revision,
            add_prefix_space=add_prefix_space,
        )

        self.model = MultiTaskModel(
            self.model_args.model_name_or_path,
            tasks=self.subtasks,
            label_names=self.label_names,
            from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
            config=self.config,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
        )

    def _create_b_to_i_label_map(self) -> Dict:
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
        return b_to_i_label

    def _transform_labels_to_ids(self, examples, tokenized_inputs) -> Tuple[List, List]:
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

                    label_ids.append(self.label_to_ids[w_label])

                # For the other tokens in a word, we set the label to either the
                # current label or -100, depending on the label_all_tokens flag.
                else:
                    if self.data_args.label_all_tokens:
                        find_dot = label[word_idx].find(".")
                        if find_dot > 0:
                            w_label = label[word_idx][:find_dot]
                        else:
                            w_label = label[word_idx]

                        label_ids.append(
                            self.b_to_i_label[st][self.label_to_ids[w_label]]
                        )
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
            task_ids.append(task_id)
        return labels, task_ids

    def _preprocess_function(self, examples):
        padding = False
        # Tokenize the texts
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            padding="max_length" if self.data_args.pad_to_max_length else False,
            max_length=self.data_args.max_seq_length,
            truncation=True,
            is_split_into_words=True,
        )

        labels, task_ids = self._transform_labels_to_ids(examples, tokenized_inputs)
        tokenized_inputs["labels"] = labels
        tokenized_inputs["task_ids"] = task_ids
        return tokenized_inputs

    def _apply_preprocessing(self) -> None:
        self.data_args.label_all_tokens = True
        self.b_to_i_label = self._create_b_to_i_label_map()
        self.label_to_ids = {
            l: i for st in self.subtasks for i, l in enumerate(self.label_names[st])
        }
        # Warn if sequence length choice is not logical
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

        if self.train_args.do_train:
            if self.data_args.max_train_samples is not None:
                max_train_samples = min(
                    len(self.raw_datasets["train"]), self.data_args.max_train_samples
                )
                self.raw_datasets["train"] = self.raw_datasets["train"].select(
                    range(max_train_samples)
                )

            with self.train_args.main_process_first(
                desc="train dataset map pre-processing"
            ):
                self.train_dataset = self.raw_datasets["train"].map(
                    self._preprocess_function,
                    batched=True,
                    load_from_cache_file=not self.data_args.overwrite_cache,
                    num_proc=self.data_args.preprocessing_num_workers,
                    desc="Running tokenizer on train dataset",
                    remove_columns=["tokens", "tags", "subtask"],
                )

        if self.train_args.do_eval:
            if self.data_args.max_eval_samples is not None:
                max_eval_samples = min(
                    len(self.raw_datasets["validation"]),
                    self.data_args.max_eval_samples,
                )
                self.raw_datasets["validation"] = self.raw_datasets[
                    "validation"
                ].select(range(max_eval_samples))
            with self.train_args.main_process_first(
                desc="validation dataset map pre-processing"
            ):
                self.eval_dataset = self.raw_datasets["validation"].map(
                    self._preprocess_function,
                    batched=True,
                    load_from_cache_file=not self.data_args.overwrite_cache,
                    num_proc=self.data_args.preprocessing_num_workers,
                    desc="Running tokenizer on validation dataset",
                    remove_columns=["tokens", "tags", "subtask"],
                )

        if self.train_args.do_predict:
            if self.data_args.max_predict_samples is not None:
                max_predict_samples = min(
                    len(self.raw_datasets["test"]), self.data_args.max_predict_samples
                )
                self.raw_datasets["test"] = self.raw_datasets["test"].select(
                    range(max_predict_samples)
                )
            with self.train_args.main_process_first(
                desc="prediction dataset map pre-processing"
            ):
                self.predict_dataset = self.raw_datasets["test"].map(
                    self._preprocess_function,
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
            else DataCollatorForTokenClassification(
                self.tokenizer,
                pad_to_multiple_of=8 if self.train_args.fp16 else None,
            )
        )

    def _retransform_labels(self, predictions, labels) -> Dict:
        predictions = predictions[0] if isinstance(predictions, tuple) else predictions
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = {st: [] for st in self.subtasks}
        true_labels = {st: [] for st in self.subtasks}
        assert len(predictions) == len(labels)
        for i, (prediction, label) in enumerate(zip(predictions, labels)):
            true_label_vec = []
            true_pred_vec = []
            # get the subtasks according to mod of index
            subtask_number = i % len(self.subtasks)
            task_name = self.subtasks[subtask_number]
            for (pr, la) in zip(prediction, label):
                if la != -100:
                    true_label = self.label_names[task_name][la]
                    true_pred = self.label_names[task_name][pr]
                    true_pred_vec.append(true_pred)
                    true_label_vec.append(true_label)
            true_predictions[task_name].append(true_pred_vec)
            true_labels[task_name].append(true_label_vec)
        return true_predictions, true_labels

    def _set_metrics(self) -> None:
        self.train_args.metric_for_best_model = "macro_f1"
        self.train_args.greater_is_better = True

    def _compute_metrics(self, p: EvalPrediction):
        predictions, labels = self._retransform_labels(p.predictions, p.label_ids)
        per_tasks_metrics = {}

        for st in self.subtasks:
            m = {}
            for average_mode in ["micro", "macro"]:
                p, r, f, _ = seqeval_metrics.precision_recall_fscore_support(
                    labels[st],
                    predictions[st],
                    average=average_mode,
                )
                m[f"{average_mode}_precision"] = p
                m[f"{average_mode}_recall"] = r
                m[f"{average_mode}_f1"] = f
            m["accuracy"] = seqeval_metrics.accuracy_score(
                labels[st],
                predictions[st],
            )
            per_tasks_metrics[st] = m

        return_metrics = {
            "accuracy": np.mean(
                [per_tasks_metrics[st]["accuracy"] for st in self.subtasks]
            ),
        }

        for st in self.subtasks:
            # accuracy per task
            return_metrics[f"{st}_accuracy"] = per_tasks_metrics[st]["accuracy"]
            for avg_mode in ["micro", "macro"]:
                for metric in ["precision", "recall", "f1"]:
                    # average metric over tasks
                    m_key = f"{avg_mode}_{metric}"
                    return_metrics[m_key] = np.mean(
                        [per_tasks_metrics[st][m_key] for st in self.subtasks]
                    )
                    # metric per task
                    m_name = f"{st}_{avg_mode}_{metric}"
                    return_metrics[m_name] = per_tasks_metrics[st][m_key]

        return return_metrics

    def _run_train_loop(self) -> None:
        # Initialize the Trainer
        self.trainer = Trainer(
            model=self.model,
            args=self.train_args,
            train_dataset=self.train_dataset if self.train_args.do_train else None,
            eval_dataset=self.eval_dataset if self.train_args.do_eval else None,
            compute_metrics=self._compute_metrics,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.model_args.early_stopping_patience
                )
            ]
            if self.model_args.early_stopping_patience
            else None,
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
            self.logger.info("*** Evaluate ***")
            metrics = self.trainer.evaluate(eval_dataset=self.eval_dataset)
            metrics["eval_samples"] = len(self.eval_dataset)

            self.trainer.log_metrics("eval", metrics)
            self.trainer.save_metrics("eval", metrics)

        # Prediction
        if self.train_args.do_predict:
            self.logger.info("*** Predict ***")
            predictions, labels, metrics = self.trainer.predict(
                self.predict_dataset, metric_key_prefix="predict"
            )
            metrics["predict_samples"] = len(self.predict_dataset)

            self.trainer.log_metrics("predict", metrics)
            self.trainer.save_metrics("predict", metrics)

            predictions_per_task, labels_per_task = self._retransform_labels(
                predictions, labels
            )

            # assemble predictions into dictionary for dumping
            prediction_dump = []
            running_index = 0
            for i, task in enumerate(self.subtasks):
                for input_text, gold_label, predicted_label in zip(
                    self.raw_datasets["test"]["tokens"][i :: len(self.subtasks)],
                    labels_per_task[task],
                    predictions_per_task[task],
                ):
                    prediction_dump.append(
                        {
                            "id": running_index,
                            "task": task,
                            "text": input_text,
                            "gold_label": gold_label,
                            "predicted_label": predicted_label,
                        }
                    )
                    running_index += 1

            # dump prediction outputs
            if self.trainer.is_world_process_zero():
                with open(
                    os.path.join(self.train_args.output_dir, "predictions.json"), "w"
                ) as output_file_stream:
                    json.dump(prediction_dump, output_file_stream, indent=4)
