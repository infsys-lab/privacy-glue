#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .policy_ie_a import file_mapping
from typing import cast
import datasets


def load_policy_ie_b(directory: str) -> datasets.DatasetDict:
    # initialize DatasetDict object
    combined = datasets.DatasetDict()

    # load tokens which are common for all sub-tasks
    tokens = datasets.load_dataset(
        "text", data_files=file_mapping(directory, "seq.in")
    ).map(lambda example: {"tokens": example["text"].split()}, remove_columns=["text"])

    # since this is task B, load all NER tags
    ner_tags_first = datasets.load_dataset(
        "text", data_files=file_mapping(directory, "seq_type_I.out")
    ).map(
        lambda example: {"ner_tags_type_one": example["text"].split()},
        remove_columns=["text"],
    )
    ner_tags_second = datasets.load_dataset(
        "text", data_files=file_mapping(directory, "seq_type_II.out")
    ).map(
        lambda example: {"ner_tags_type_two": example["text"].split()},
        remove_columns=["text"],
    )

    # mypy-related fixes
    tokens = cast(datasets.DatasetDict, tokens)
    ner_tags_first = cast(datasets.DatasetDict, ner_tags_first)
    ner_tags_second = cast(datasets.DatasetDict, ner_tags_second)

    # zip together data in splits
    for split in ["train", "validation", "test"]:
        combined[split] = datasets.concatenate_datasets(
            [tokens[split], ner_tags_first[split], ner_tags_second[split]], axis=1
        )

    # merge NER tags and drop old ones
    combined = combined.map(
        lambda x: {
            "ner_tags": list(zip(x["ner_tags_type_one"], x["ner_tags_type_two"]))
        },
        remove_columns=["ner_tags_type_one", "ner_tags_type_two"],
    )

    return combined
