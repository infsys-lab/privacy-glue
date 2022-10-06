#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import logging
import torch
import torch.nn as nn
import numpy as np
import evaluate
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    PreTrainedModel,
    PretrainedConfig,
)

from parser import DataArguments, ModelArguments
from utils.pipeline_utils import Privacy_GLUE_Pipeline

logger = logging.getLogger(__name__)

TASK2SUBTASKS = {
    "piextract": {st: [st] for st in ["SHARE", "COLLECT", "NOT_COLLECT", "NOT_SHARE"]},
    "policy_ie_b": {
        "type-I": [
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
        "type-II": ["purpose-argument", "polarity", "method", "condition-argument"],
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
        self.subtasks = TASK2SUBTASKS[data_args.task].keys()

    def _retrieve_data(self) -> None:
        data = self._get_data()

        if self.train_args.do_train:
            self.train_dataset = data["train"]
        if self.train_args.do_eval:
            self.eval_dataset = data["validation"]

        if self.train_args.do_predict:
            self.predict_dataset = data["test"]

    def _load_pretrained_model_and_tokenizer(self) -> None:
        # Distributed training:
        # The .from_pretrained methods guarantee that only one local process
        # can concurrently download model & vocab.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.tokenizer_name
            if self.model_args.tokenizer_name
            else self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            use_fast=True,
            revision=self.model_args.model_revision,
        )
        self.subtask_models = {}
        for subtask in self.subtasks:
            label_names = ["O"] + [
                f"{pre}-{label}"
                for pre in ["B", "I"]
                for label in TASK2SUBTASKS[self.data_args.task][subtask]
            ]
            subtask_config = AutoConfig.from_pretrained(
                self.model_args.model_name_or_path,
                finetuning_task=self.data_args.task,
                cache_dir=self.model_args.cache_dir,
                revision=self.model_args.model_revision,
                problem_type="singlelabel_classification",
                num_labels=len(label_names),
                id2label=dict(enumerate(label_names)),
                label2id={l: n for n, l in enumerate(label_names)},
            )

            subtask_model = AutoModelForTokenClassification.from_pretrained(
                self.model_args.model_name_or_path,
                from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
                config=subtask_config,
                cache_dir=self.model_args.cache_dir,
                revision=self.model_args.model_revision,
            )
            self.subtask_models[subtask] = subtask_model
        self.model = MultitaskModel("bert", self.subtask_models)

    def _apply_preprocessing(self, subtask) -> None:
        padding = False
        # Map that sends B-Xxx label to its I-Xxx counterpart
        b_to_i_label = []
        for idx, label in enumerate(self.label_names[subtask]):
            if label.startswith("B-") and label.replace("B-", "I-") in self.label_names:
                b_to_i_label.append(self.label_names.index(label.replace("B-", "I-")))
            else:
                b_to_i_label.append(idx)

        def preprocess_function(examples):
            # Tokenize the texts
            tokenized_inputs = self.tokenizer(
                examples["tokens"],
                padding=padding,
                truncation=True,
                is_split_into_words=True,
            )
            labels = []
            for i, multilabel in enumerate(examples["ner_tags"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                label_ids = []
                transformed_label = [-1.0 for _ in self.label_names]
                assert len(transformed_label) == 9
                for word_idx in word_ids:
                    # Special tokens have a word id that is None.
                    # We set the label to -100 so they are automatically
                    # ignored in the loss function.
                    if word_idx is None:
                        label_ids.append([-100 for _ in range(6)])
                    else:
                        for label in multilabel[word_idx]:
                            transformed_label[self.config.label2id[label]] = 1.0
                    label_ids.append(transformed_label)
                labels.append(label_ids)
            tokenized_inputs["labels"] = labels
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
                    desc="Running tokenizer on train dataset",
                    remove_columns=["tokens"],
                )
                self.train_dataset.set_format("torch")

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
                    remove_columns=["tokens"],
                )
                self.eval_dataset.set_format("torch")

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
                    remove_columns=["tokens"],
                )
                self.predict_dataset.set_format("torch")
        breakpoint()

    def _set_metrics(self) -> None:
        # Get the metric function
        metric = evaluate.load("f1", "multilabel")
        # using sample f1
        averaging = "samples"

        def transform_to_binary(pred):
            return np.round(torch.special.expit(torch.Tensor(pred)))

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

    def _run_train_loop(self) -> None:
        # Initialize the Trainer
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


class MultitaskModel(PreTrainedModel):
    def __init__(self, encoder, taskmodels_dict):
        """
        Setting MultitaskModel up as a PretrainedModel allows us
        to take better advantage of Trainer features
        """
        super().__init__(PretrainedConfig())

        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def create(cls, model_dict):
        """
        This creates a MultitaskModel using the model class and config objects
        from single-task models.

        We do this by creating each single-task model, and having them share
        the same encoder transformer.
        """
        shared_encoder = None
        taskmodels_dict = {}
        for task_name, model in model_dict.items():
            if shared_encoder is None:
                shared_encoder = getattr(model, cls.get_encoder_attr_name(model))
            else:
                setattr(model, cls.get_encoder_attr_name(model), shared_encoder)
            taskmodels_dict[task_name] = model
        return cls(encoder=shared_encoder, taskmodels_dict=taskmodels_dict)

    @classmethod
    def get_encoder_attr_name(cls, model):
        """
        The encoder transformer is named differently in each model "architecture".
        This method lets us get the name of the encoder attribute
        """
        model_class_name = model.__class__.__name__
        if model_class_name.startswith("Bert"):
            return "bert"
        else:
            raise KeyError(f"Add support for new model {model_class_name}")

    def forward(self, task_name, **kwargs):
        return self.taskmodels_dict[task_name](**kwargs)
