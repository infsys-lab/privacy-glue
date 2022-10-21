#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from transformers import Trainer
from transformers.trainer_utils import PredictionOutput


class QuestionAnsweringTrainer(Trainer):
    """
    Subclass of `Trainer` specific to Question-Answering tasks
    Post-processing utilities for question answering.
    """

    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def evaluate(
        self,
        eval_dataset=None,
        eval_examples=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = (
            self.prediction_loop
            if self.args.use_legacy_prediction_loop
            else self.evaluation_loop
        )
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise
                # we defer to self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(
                eval_examples, eval_dataset, output.predictions
            )
            metrics = self.compute_metrics(eval_preds)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)
        else:
            metrics = {}

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        return metrics

    def predict(
        self,
        predict_dataset,
        predict_examples,
        ignore_keys=None,
        metric_key_prefix: str = "predict",
    ):
        predict_dataloader = self.get_test_dataloader(predict_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = (
            self.prediction_loop
            if self.args.use_legacy_prediction_loop
            else self.evaluation_loop
        )
        try:
            output = eval_loop(
                predict_dataloader,
                description="Prediction",
                # No point gathering the predictions if there are no metrics,
                # otherwise we defer to self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        predictions = self.post_process_function(
            predict_examples,
            predict_dataset,
            output.predictions,
        )
        metrics = self.compute_metrics(predictions)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return PredictionOutput(
            predictions=predictions.predictions,
            label_ids=predictions.label_ids,
            metrics=metrics,
        )


class Weighted_Random_Sampler_Trainer(Trainer):
    """
    Subclass of Trainer which overrides the get_train_loader function to
    provide a data loader which samples from the data with a weighted sampler
    to counteract the very unbalanced datasets
    """

    def _get_sample_weights(self):
        train_dataset_labels = self.train_dataset["label"]
        unique_labels = sorted(set(train_dataset_labels))
        samples_count = [train_dataset_labels.count(label) for label in unique_labels]
        samples_weight = 1 / torch.Tensor(samples_count)
        samples_weight = torch.Tensor(
            [
                samples_weight[unique_labels.index(label)]
                for label in train_dataset_labels
            ]
        )
        return samples_weight

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        data_collator = self.data_collator
        samples_weight = self._get_sample_weights()
        train_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
        )
