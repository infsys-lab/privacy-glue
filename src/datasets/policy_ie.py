#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datasets
import os


def load_policy_ie(directory: str) -> datasets.DatasetDict:
    data_files = {}
    data_files["train"] = os.path.join(directory, "train", "seq.in")
    data_files["validation"] = os.path.join(directory, "valid", "seq.in")
    data_files["test"] = os.path.join(directory, "test", "seq.in")
    tokens = datasets.load_dataset("text", data_files=data_files).map(
        lambda example: {"tokens": example["text"].split()},
        remove_columns=["text"])
    data_files["train"] = os.path.join(directory, "train", "label")
    data_files["validation"] = os.path.join(directory, "valid", "label")
    data_files["test"] = os.path.join(directory, "test", "label")
    labels = datasets.load_dataset("text",
                                   data_files=data_files).rename_column(
                                       "text", "label")
    data_files["train"] = os.path.join(directory, "train", "seq_type_I.out")
    data_files["validation"] = os.path.join(directory, "valid",
                                            "seq_type_I.out")
    data_files["test"] = os.path.join(directory, "test", "seq_type_I.out")
    ner_tags_first = datasets.load_dataset("text", data_files=data_files).map(
        lambda example: {"ner_tags_type_one": example["text"].split()},
        remove_columns=["text"])
    data_files["train"] = os.path.join(directory, "train", "seq_type_II.out")
    data_files["validation"] = os.path.join(directory, "valid",
                                            "seq_type_II.out")
    data_files["test"] = os.path.join(directory, "test", "seq_type_II.out")
    ner_tags_second = datasets.load_dataset("text", data_files=data_files).map(
        lambda example: {"ner_tags_type_two": example["text"].split()},
        remove_columns=["text"])
    combined = datasets.DatasetDict()
    for (split, a), (_, b), (_, c), (_, d) in zip(tokens.items(),
                                                  labels.items(),
                                                  ner_tags_first.items(),
                                                  ner_tags_second.items()):
        combined[split] = datasets.concatenate_datasets([a, b, c, d], axis=1)
    return combined
