#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
import numpy as np
import evaluate
import torch
import os


class Sequence_Classification_Pipeline(Privacy_GLUE_Pipeline):
    """
    Subclass of abstract Privacy_GLUE_Pipeline class with implementations
    of task specific pipeline functions.
    """

    # define constant class variables
    task2problem = {
        "opp_115": "multi_label",
        "policy_detection": "single_label",
        "policy_ie_a": "single_label",
        "privacy_qa": "single_label",
    }
    task2input = {
        "opp_115": ["text"],
        "policy_detection": ["text"],
        "policy_ie_a": ["text"],
        "privacy_qa": ["question", "text"],
    }

    def __init__(
        self,
        data_args: DataArguments,
        model_args: ModelArguments,
        training_args: TrainingArguments,
    ) -> None:
        # initialize parent class and add new attributes
        super().__init__(data_args, model_args, training_args)
        self.problem_type = self.task2problem[self.data_args.task]
        self.input_keys = self.task2input[self.data_args.task]

    def _retrieve_data(self) -> None:
        # load data and label names
        data = self._get_data()
        self.label_names = (
            data["train"].features["label"].names
            if self.problem_type == "single_label"
            else data["train"].features["label"].feature.names
        )

        if self.train_args.do_train:
            self.train_dataset = data["train"]

        if self.train_args.do_eval:
            self.eval_dataset = data["validation"]

        if self.train_args.do_predict:
            self.predict_dataset = data["test"]

    def _load_pretrained_model_and_tokenizer(self) -> None:
        # load model config
        self.config = AutoConfig.from_pretrained(
            self.model_args.config_name
            if self.model_args.config_name
            else self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
            problem_type=f"{self.problem_type}_classification",
            num_labels=len(self.label_names),
            id2label=dict(enumerate(self.label_names)),
            label2id={l: n for n, l in enumerate(self.label_names)},
        )

        # load model tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.tokenizer_name
            if self.model_args.tokenizer_name
            else self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            use_fast=self.model_args.use_fast_tokenizer,
            revision=self.model_args.model_revision,
        )

        # load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_args.model_name_or_path,
            from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
            config=self.config,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
        )

    def _preprocess_function(self, examples):
        # load input text from relevant keys
        input_text = tuple([examples[key] for key in self.input_keys])

        # tokenize the texts
        tokenized_examples = self.tokenizer(
            *input_text,
            padding="max_length" if self.data_args.pad_to_max_length else False,
            max_length=self.max_seq_length,
            truncation=True,
        )

        # convert labels to multi-hot vector if multi-label
        if self.problem_type == "multi_label":
            tokenized_examples["labels"] = [
                [
                    1.0 if index in example else 0.0
                    for index, _ in enumerate(self.label_names)
                ]
                for example in examples["label"]
            ]

        return tokenized_examples

    def _apply_preprocessing(self) -> None:
        # warn if sequence length choice is not logical
        if self.data_args.max_seq_length > self.tokenizer.model_max_length:
            self.logger.warning(
                f"The max_seq_length passed ({self.data_args.max_seq_length}) "
                "is larger than the maximum length for the "
                f"model ({self.tokenizer.model_max_length}). "
                f"Using max_seq_length={self.tokenizer.model_max_length}"
            )

        # use logic to choose maximum sequence length
        self.max_seq_length = min(
            self.data_args.max_seq_length, self.tokenizer.model_max_length
        )

        # proceed to preprocess train dataset
        if self.train_args.do_train:
            # subsample if necessary
            if self.data_args.max_train_samples is not None:
                max_train_samples = min(
                    len(self.train_dataset), self.data_args.max_train_samples
                )
                self.train_dataset = self.train_dataset.select(range(max_train_samples))

            # batch-apply the preprocessing function
            with self.train_args.main_process_first(
                desc="train dataset map pre-processing"
            ):
                self.train_dataset = self.train_dataset.map(
                    self._preprocess_function,
                    batched=True,
                    num_proc=self.data_args.preprocessing_num_workers,
                    load_from_cache_file=not self.data_args.overwrite_cache,
                    desc="Running tokenizer on train dataset",
                    remove_columns=self.input_keys
                    if self.problem_type == "single_label"
                    else self.input_keys + ["label"],
                )

        # proceed to preprocess eval dataset
        if self.train_args.do_eval:
            # subsample if necessary
            if self.data_args.max_eval_samples is not None:
                max_eval_samples = min(
                    len(self.eval_dataset), self.data_args.max_eval_samples
                )
                self.eval_dataset = self.eval_dataset.select(range(max_eval_samples))

            # batch-apply the preprocessing function
            with self.train_args.main_process_first(
                desc="validation dataset map pre-processing"
            ):
                self.eval_dataset = self.eval_dataset.map(
                    self._preprocess_function,
                    batched=True,
                    num_proc=self.data_args.preprocessing_num_workers,
                    load_from_cache_file=not self.data_args.overwrite_cache,
                    desc="Running tokenizer on validation dataset",
                    remove_columns=self.input_keys
                    if self.problem_type == "single_label"
                    else self.input_keys + ["label"],
                )

        # proceed to preprocess predict dataset
        if self.train_args.do_predict:
            if self.data_args.max_predict_samples is not None:
                # subsample if necessary
                max_predict_samples = min(
                    len(self.predict_dataset), self.data_args.max_predict_samples
                )
                self.predict_dataset = self.predict_dataset.select(
                    range(max_predict_samples)
                )

            # batch-apply the preprocessing function
            with self.train_args.main_process_first(
                desc="prediction dataset map pre-processing"
            ):
                self.predict_dataset = self.predict_dataset.map(
                    self._preprocess_function,
                    batched=True,
                    num_proc=self.data_args.preprocessing_num_workers,
                    load_from_cache_file=not self.data_args.overwrite_cache,
                    desc="Running tokenizer on prediction dataset",
                    remove_columns=self.input_keys
                    if self.problem_type == "single_label"
                    else self.input_keys + ["label"],
                )

        # define the data collator that we will use
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

        # get the metric function
        if self.problem_type == "multi_label":
            metric = evaluate.load("f1", "multilabel")
            averaging = "samples"

            # define inner function
            def transform_to_binary(pred):
                return np.round(torch.special.expit(torch.Tensor(pred)))

        else:
            metric = evaluate.load("f1", "multiclass")
            averaging = "macro"

            # define inner function
            def transform_to_binary(pred):
                return np.argmax(pred, axis=1)

        # define outer function to use
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

        # assign this function as an attribute
        self.compute_metrics = compute_f1

    def _run_train_loop(self) -> None:
        # conditionally choose the trainer class
        if self.data_args.task == "privacy_qa":
            this_task_trainer = Weighted_Random_Sampler_Trainer
        else:
            this_task_trainer = Trainer

        # initialize trainer with arguments
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

        # execute training
        if self.train_args.do_train:
            train_result = self.trainer.train(
                resume_from_checkpoint=self.last_checkpoint
            )
            metrics = train_result.metrics
            metrics["train_samples"] = len(self.train_dataset)
            self.trainer.save_model()
            self.trainer.log_metrics("train", metrics)
            self.trainer.save_metrics("train", metrics)
            self.trainer.save_state()

        # execute evaluation
        if self.train_args.do_eval:
            self.logger.info("*** Evaluate ***")
            metrics = self.trainer.evaluate(eval_dataset=self.eval_dataset)
            metrics["eval_samples"] = len(self.eval_dataset)
            self.trainer.log_metrics("eval", metrics)
            self.trainer.save_metrics("eval", metrics)

        # execute prediction
        if self.train_args.do_predict:
            self.logger.info("*** Predict ***")
            predictions, labels, metrics = self.trainer.predict(
                self.predict_dataset, metric_key_prefix="predict"
            )
            metrics["predict_samples"] = len(self.predict_dataset)
            self.trainer.log_metrics("predict", metrics)
            self.trainer.log(metrics)
            self.trainer.save_metrics("predict", metrics)

            # convert integer/float labels to more readable format
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

            # dump prediction outputs
            output_predict_file = os.path.join(
                self.train_args.output_dir, "predictions.txt"
            )
            if self.trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    writer.write("index\tprediction\ttrue_label\n")
                    for index, (item, true_l) in enumerate(zip(predictions, labels)):
                        writer.write(f"{index}\t{item}\t{true_l}\n")
