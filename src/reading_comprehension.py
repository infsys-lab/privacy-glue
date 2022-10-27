#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import json
import os
from typing import Tuple

import evaluate
import numpy as np
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EvalPrediction,
    PreTrainedTokenizerFast,
    default_data_collator,
)

from utils.pipeline_utils import Privacy_GLUE_Pipeline
from utils.trainer_utils import QuestionAnsweringTrainer


class Reading_Comprehension_Pipeline(Privacy_GLUE_Pipeline):
    def _retrieve_data(self) -> None:
        self.raw_datasets = self._get_data()

    def _load_pretrained_model_and_tokenizer(self):
        # Load pretrained model and tokenizer
        # in distributed training, the .from_pretrained methods guarantee that only one
        # local process can concurrently download model & vocab.
        self.config = AutoConfig.from_pretrained(
            self.model_args.config_name
            if self.model_args.config_name
            else self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.tokenizer_name
            if self.model_args.tokenizer_name
            else self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            use_fast=True,
            revision=self.model_args.model_revision,
        )
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            self.model_args.model_name_or_path,
            from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
            config=self.config,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
        )

        # Tokenizer check: this script requires a fast tokenizer
        if not isinstance(self.tokenizer, PreTrainedTokenizerFast):
            raise ValueError(
                "This script only works for models that have a fast tokenizer. "
                "Check out the big table of models at "
                "https://huggingface.co/transformers/index.html#supported-frameworks "
                "to find the model types that meet this requirement"
            )

    def _prepare_train_features(self, examples):
        # Some of the questions have lots of whitespace on the left,
        # which is not useful and will make the truncation of the context fail
        # (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[self.question_column_name] = [
            q.lstrip() for q in examples[self.question_column_name]
        ]

        # Tokenize our examples with truncation and maybe padding, but keep the
        # overflows using a stride. This results in one example possible
        # giving several features when a context is long, each of those features
        # having a context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples[
                self.question_column_name
                if self.pad_on_right
                else self.context_column_name
            ],
            examples[
                self.context_column_name
                if self.pad_on_right
                else self.question_column_name
            ],
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.max_seq_length,
            stride=self.data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if self.data_args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long
        # context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # The offset mappings will give us a map from token to character position
        # in the original context. This will help us compute the start_positions
        # and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        # Now loop through them
        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example
            # (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example
            # containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[self.answer_column_name][sample_index]

            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (
                    1 if self.pad_on_right else 0
                ):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if self.pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this
                # feature is labeled with the CLS index).
                if not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to
                    # the two ends of the answer. Note: we could go after the
                    # last offset if the answer is the last word (edge case).
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    def _prepare_validation_features(self, examples):
        # Some of the questions have lots of whitespace on the left,
        # which is not useful and will make the truncation of the context fail
        # (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[self.question_column_name] = [
            q.lstrip() for q in examples[self.question_column_name]
        ]

        # Tokenize our examples with truncation and maybe padding, but keep the
        # overflows using a stride. This results in one example possible giving
        # several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples[
                self.question_column_name
                if self.pad_on_right
                else self.context_column_name
            ],
            examples[
                self.context_column_name
                if self.pad_on_right
                else self.question_column_name
            ],
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.max_seq_length,
            stride=self.data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if self.data_args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long
        # context, we need a map from a feature to its corresponding example.
        # This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings
        # of the context, so we keep the corresponding example_id and we will
        # store the offset mappings.
        tokenized_examples["example_id"] = []

        # Iterate over examples
        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example
            # (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if self.pad_on_right else 0

            # One example can give several spans, this is the index of the example
            # containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so
            # it's easy to determine if a token position is part of the context
            # or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    def _apply_preprocessing(self) -> None:
        # Preprocessing is slighlty different for training and evaluation.
        if self.train_args.do_train:
            column_names = self.raw_datasets["train"].column_names
        elif self.train_args.do_eval:
            column_names = self.raw_datasets["validation"].column_names
        else:
            column_names = self.raw_datasets["test"].column_names
        self.question_column_name = (
            "question" if "question" in column_names else column_names[0]
        )
        self.context_column_name = (
            "context" if "context" in column_names else column_names[1]
        )
        self.answer_column_name = (
            "answers" if "answers" in column_names else column_names[2]
        )

        # Padding side determines if we do (question|context) or (context|question).
        self.pad_on_right = self.tokenizer.padding_side == "right"

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

        # Preprocess training data
        if self.train_args.do_train:
            if self.data_args.max_train_samples is not None:
                # We will select sample from whole data if argument is specified
                max_train_samples = min(
                    len(self.raw_datasets["train"]), self.data_args.max_train_samples
                )
                self.raw_datasets["train"] = self.raw_datasets["train"].select(
                    range(max_train_samples)
                )

            # Create train feature from dataset
            with self.train_args.main_process_first(
                desc="train dataset map pre-processing"
            ):
                self.train_dataset = self.raw_datasets["train"].map(
                    self._prepare_train_features,
                    batched=True,
                    num_proc=self.data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not self.data_args.overwrite_cache,
                    desc="Running tokenizer on train dataset",
                )

            if self.data_args.max_train_samples is not None:
                # Number of samples might increase during Feature Creation
                # we select only specified max samples
                max_train_samples = min(
                    len(self.train_dataset), self.data_args.max_train_samples
                )
                self.train_dataset = self.train_dataset.select(range(max_train_samples))

        # Preprocess evaluation data
        if self.train_args.do_eval:
            if self.data_args.max_eval_samples is not None:
                # We will select sample from whole data
                max_eval_samples = min(
                    len(self.raw_datasets["validation"]),
                    self.data_args.max_eval_samples,
                )
                self.raw_datasets["validation"] = self.raw_datasets[
                    "validation"
                ].select(range(max_eval_samples))

            # Validation Feature Creation
            with self.train_args.main_process_first(
                desc="validation dataset map pre-processing"
            ):
                self.eval_dataset = self.raw_datasets["validation"].map(
                    self._prepare_validation_features,
                    batched=True,
                    num_proc=self.data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not self.data_args.overwrite_cache,
                    desc="Running tokenizer on validation dataset",
                )

            if self.data_args.max_eval_samples is not None:
                # During Feature creation dataset samples might increase,
                # we will select required samples again
                max_eval_samples = min(
                    len(self.eval_dataset), self.data_args.max_eval_samples
                )
                self.eval_dataset = self.eval_dataset.select(range(max_eval_samples))

        # Preprocess prediction data
        if self.train_args.do_predict:
            if self.data_args.max_predict_samples is not None:
                # We will select sample from whole data
                max_predict_samples = min(
                    len(self.raw_datasets["test"]),
                    self.data_args.max_predict_samples,
                )
                self.raw_datasets["test"] = self.raw_datasets["test"].select(
                    range(max_predict_samples)
                )

            # Predict Feature Creation
            with self.train_args.main_process_first(
                desc="prediction dataset map pre-processing"
            ):
                self.predict_dataset = self.raw_datasets["test"].map(
                    self._prepare_validation_features,
                    batched=True,
                    num_proc=self.data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not self.data_args.overwrite_cache,
                    desc="Running tokenizer on prediction dataset",
                )

            if self.data_args.max_predict_samples is not None:
                # During Feature creation dataset samples might increase,
                # we will select required samples again
                max_predict_samples = min(
                    len(self.predict_dataset), self.data_args.max_predict_samples
                )
                self.predict_dataset = self.predict_dataset.select(
                    range(max_predict_samples)
                )

        # Data collator: we have already padded to max length if the
        # corresponding flag is True, otherwise we need to pad in the data
        # collator.
        self.data_collator = (
            default_data_collator
            if self.data_args.pad_to_max_length
            else DataCollatorWithPadding(
                self.tokenizer,
                pad_to_multiple_of=8 if self.train_args.fp16 else None,
            )
        )

    def _postprocess_qa_predictions(
        self,
        examples,
        features,
        predictions: Tuple[np.ndarray, np.ndarray],
    ):
        if len(predictions) != 2:
            raise ValueError(
                "`predictions` should be a tuple with two elements "
                "(start_logits, end_logits)."
            )
        if len(predictions[0]) != len(features):
            raise ValueError(
                f"Got {len(predictions[0])} predictions and {len(features)} features."
            )

        # Build a map example to its corresponding features.
        all_start_logits, all_end_logits = predictions
        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        features_per_example = collections.defaultdict(list)
        for i, feature in enumerate(features):
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)

        # The dictionaries we have to fill.
        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()

        # Logging.
        self.logger.info(
            f"Post-processing {len(examples)} example predictions "
            f"split into {len(features)} features."
        )

        # Let's loop over all the examples!
        for example_index, example in enumerate(tqdm(examples)):
            # Those are the indices of the features associated to the current example.
            feature_indices = features_per_example[example_index]
            prelim_predictions = []

            # Looping through all the features associated to the current example.
            for feature_index in feature_indices:
                # We grab the predictions of the model for this feature.
                start_logits = all_start_logits[feature_index]
                end_logits = all_end_logits[feature_index]
                # This is what will allow us to map some the positions in our logits
                # to span of texts in the original context.
                offset_mapping = features[feature_index]["offset_mapping"]

                # Go through all possibilities for the `n_best_size` greater
                # start and end logits.
                start_indexes = np.argsort(start_logits)[
                    -1 : -self.data_args.n_best_size - 1 : -1
                ].tolist()
                end_indexes = np.argsort(end_logits)[
                    -1 : -self.data_args.n_best_size - 1 : -1
                ].tolist()

                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Don't consider out-of-scope answers, either because the
                        # indices are out of bounds or correspond
                        # to part of the input_ids that are not in the context.
                        if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or len(offset_mapping[start_index]) < 2
                            or offset_mapping[end_index] is None
                            or len(offset_mapping[end_index]) < 2
                        ):
                            continue  # pragma: no cover
                        # Don't consider answers with a length that is either
                        # < 0 or > max_answer_length.
                        elif (
                            end_index < start_index
                            or end_index - start_index + 1
                            > self.data_args.max_answer_length
                        ):
                            continue  # pragma: no cover
                        else:
                            prelim_predictions.append(
                                {
                                    "offsets": (
                                        offset_mapping[start_index][0],
                                        offset_mapping[end_index][1],
                                    ),
                                    "score": start_logits[start_index]
                                    + end_logits[end_index],
                                    "start_logit": start_logits[start_index],
                                    "end_logit": end_logits[end_index],
                                }
                            )

            # Only keep the best `n_best_size` predictions.
            predictions = sorted(
                prelim_predictions, key=lambda x: x["score"], reverse=True
            )[: self.data_args.n_best_size]

            # Use the offsets to gather the answer text in the original context.
            context = example["context"]
            for pred in predictions:
                offsets = pred.pop("offsets")
                pred["text"] = context[offsets[0] : offsets[1]]

            # In the very rare edge case we have not a single non-null prediction,
            # we create a fake prediction to avoid failure.
            if len(predictions) == 0 or (
                len(predictions) == 1 and predictions[0]["text"] == ""
            ):
                predictions.insert(
                    0,
                    {
                        "text": "empty",
                        "start_logit": 0.0,
                        "end_logit": 0.0,
                        "score": 0.0,
                    },
                )

            # Compute the softmax of all scores (we do it with numpy to stay independent
            # from torch/tf in this file, using the LogSumExp trick).
            scores = np.array([pred.pop("score") for pred in predictions])
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()

            # Include the probabilities in our predictions.
            for prob, pred in zip(probs, predictions):
                pred["probability"] = prob

            # Pick the best prediction. If the null answer is not possible, this is easy
            all_predictions[example["id"]] = predictions[0]["text"]

            # Make `predictions` JSON-serializable by casting np.float back to float.
            all_nbest_json[example["id"]] = [
                {
                    k: (
                        float(v)
                        if isinstance(v, (np.float16, np.float32, np.float64))
                        else v
                    )
                    for k, v in pred.items()
                }
                for pred in predictions
            ]

        return all_predictions, all_nbest_json

    def _post_processing_function(self, examples, features, predictions):
        # Post-processing: we match the start logits and end logits to answers
        # in the original context.
        predictions, _ = self._postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
        )

        # Format the result to the format the metric expects.
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]
        references = [
            {"id": ex["id"], "answers": ex[self.answer_column_name]} for ex in examples
        ]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    def _set_metrics(self) -> None:
        self.metric = evaluate.load("squad")
        self.train_args.metric_for_best_model = "f1"
        self.train_args.greater_is_better = True

    def _compute_metrics(self, p: EvalPrediction):
        return self.metric.compute(predictions=p.predictions, references=p.label_ids)

    def _run_train_loop(self) -> None:
        # Initialize our Trainer
        self.trainer = QuestionAnsweringTrainer(
            model=self.model,
            args=self.train_args,
            train_dataset=self.train_dataset if self.train_args.do_train else None,
            eval_dataset=self.eval_dataset if self.train_args.do_eval else None,
            eval_examples=self.raw_datasets["validation"]
            if self.train_args.do_eval
            else None,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            post_process_function=self._post_processing_function,
            compute_metrics=self._compute_metrics,
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
            self.trainer.save_model()
            metrics = train_result.metrics
            metrics["train_samples"] = len(self.train_dataset)
            self.trainer.log_metrics("train", metrics)
            self.trainer.save_metrics("train", metrics)
            self.trainer.save_state()

        # Evaluation
        if self.train_args.do_eval:
            self.logger.info("*** Evaluate ***")
            metrics = self.trainer.evaluate()
            metrics["eval_samples"] = len(self.eval_dataset)
            self.trainer.log_metrics("eval", metrics)
            self.trainer.save_metrics("eval", metrics)

        # Prediction
        if self.train_args.do_predict:
            self.logger.info("*** Predict ***")
            results = self.trainer.predict(
                self.predict_dataset, self.raw_datasets["test"]
            )
            metrics = results.metrics
            metrics["predict_samples"] = len(self.predict_dataset)
            self.trainer.log_metrics("predict", metrics)
            self.trainer.save_metrics("predict", metrics)

            # create prediction dump
            for prediction in results.predictions:
                matched_row = self.raw_datasets["test"].filter(
                    lambda example: example["id"] == prediction["id"]
                )
                prediction["title"] = matched_row["title"].pop()
                prediction["context"] = matched_row["context"].pop()
                prediction["question"] = matched_row["question"].pop()
                prediction["gold_answers"] = matched_row["answers"].pop()["text"]

            # dump predicted results
            if self.trainer.is_world_process_zero():
                with open(
                    os.path.join(self.train_args.output_dir, "predictions.json"),
                    "w",
                ) as output_file_stream:
                    json.dump(results.predictions, output_file_stream)
