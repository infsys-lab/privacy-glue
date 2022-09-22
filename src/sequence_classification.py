#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import numpy as np
import evaluate
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)
from parser import DataArguments, ModelArguments

from utils.pipeline_utils import Privacy_GLUE_Pipeline

logger = logging.getLogger(__name__)

TASK2PROBLEM_TYPE = {"opp_115": "multi_label"}


class Sequence_Classification_Pipeline(Privacy_GLUE_Pipeline):
    def __init__(
        self,
        data_args: DataArguments,
        model_args: ModelArguments,
        training_args: TrainingArguments,
    ) -> None:
        super().__init__(data_args, model_args, training_args)
        self.problem_type = TASK2PROBLEM_TYPE.get(self.data_args.task, "single_label")
        if self.train_args.gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.train_args.gpu_id)

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
            config=self.config,
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
            # Tokenize the texts
            return self.tokenizer(
                examples["text"],
                padding=padding,
                # max_length=self.data_args.max_seq_length,
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
                    remove_columns=["text"],
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
                    remove_columns=["text"],
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
                    remove_columns=["text"],
                )
                self.predict_dataset.set_format("torch")

    def _set_metrics(self) -> None:
        # Get the metric function
        if self.problem_type == "multi_label":
            metric = evaluate.load("f1", "multilabel")
            averaging = "samples"

            def transform_to_binary(pred):
                return torch.special.expit(torch.Tensor(pred))

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
            args=self.train_args,  # TrainingArguments(".", num_train_epochs=1), #
            train_dataset=self.train_dataset if self.train_args.do_train else None,
            eval_dataset=self.eval_dataset if self.train_args.do_eval else None,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )
        print(f"training on cuda:{torch.cuda.current_device()}")
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


# def sequence_classification(data_args, model_args, training_args):
#     # wandb.init(mode="disabled")
#     pipeline = _Sequence_Classification_Pipeline(data_args, model_args, training_args)
#     pipeline.retrieve_data()
#     pipeline.load_pretrained_model_and_tokenizer()
#     pipeline.apply_preprocessing()
#     pipeline.set_metrics()
#     pipeline.run_train_loop()
#     pipeline.destroy()
