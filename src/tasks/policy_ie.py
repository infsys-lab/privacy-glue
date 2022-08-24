#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict
import datasets
import os


def file_mapping(directory: str, filename: str) -> Dict[str, str]:
    # define patterns for file loading
    files = {}
    files["train"] = os.path.join(directory, "train", filename)
    files["validation"] = os.path.join(directory, "valid", filename)
    files["test"] = os.path.join(directory, "test", filename)

    return files


def load_policy_ie_a(directory: str) -> datasets.DatasetDict:
    return load_policy_ie(directory, "a")


def load_policy_ie_b(directory: str) -> datasets.DatasetDict:
    return load_policy_ie(directory, "b")


def load_policy_ie(directory: str, subtype: str) -> datasets.DatasetDict:
    # initialize DatasetDict object
    combined = datasets.DatasetDict()

    # load tokens which are common for all sub-tasks
    tokens = datasets.load_dataset(
        "text", data_files=file_mapping(directory, "seq.in")).map(
            lambda example: {"tokens": example["text"].split()},
            remove_columns=["text"])

    # proceed conditionally dependent on subtask
    if subtype == "a":
        # if task A, only load labels
        labels = datasets.load_dataset(
            "text",
            data_files=file_mapping(directory,
                                    "label")).rename_column("text", "label")

        # zip together data
        for (split,
             tokens_split), (_, labels_split) in zip(tokens.items(),
                                                     labels.items()):
            combined[split] = datasets.concatenate_datasets(
                [tokens_split, labels_split], axis=1)

    elif subtype == "b":
        # if task B, load all NER tags
        ner_tags_first = datasets.load_dataset(
            "text", data_files=file_mapping(directory, "seq_type_I.out")).map(
                lambda example: {"ner_tags_type_one": example["text"].split()},
                remove_columns=["text"])
        ner_tags_second = datasets.load_dataset(
            "text", data_files=file_mapping(directory, "seq_type_II.out")).map(
                lambda example: {"ner_tags_type_two": example["text"].split()},
                remove_columns=["text"])

    return combined
