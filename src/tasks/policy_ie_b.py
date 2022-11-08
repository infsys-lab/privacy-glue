#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import cast

import datasets

from utils.task_utils import policy_ie_file_mapping
from utils.tasks_utils import expand_dataset_per_task

SUBTASKS = ["type-I", "type-II"]
LABELS = [
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
]


def load_policy_ie_b(directory: str) -> datasets.DatasetDict:
    # initialize DatasetDict object
    combined = datasets.DatasetDict()

    # load tokens which are common for all sub-tasks
    tokens = datasets.load_dataset(
        "text", data_files=policy_ie_file_mapping(directory, "seq.in")
    ).map(lambda example: {"tokens": example["text"].split()}, remove_columns=["text"])

    # since this is task B, load all NER tags
    ner_tags_first = datasets.load_dataset(
        "text", data_files=policy_ie_file_mapping(directory, "seq_type_I.out")
    ).map(
        lambda example: {"ner_tags_type_one": example["text"].split()},
        remove_columns=["text"],
    )
    ner_tags_second = datasets.load_dataset(
        "text", data_files=policy_ie_file_mapping(directory, "seq_type_II.out")
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
        lambda x: {"tags": list(zip(x["ner_tags_type_one"], x["ner_tags_type_two"]))},
        remove_columns=["ner_tags_type_one", "ner_tags_type_two"],
    )

    # reassign splits to combined and multiply tags to rows
    combined["train"] = expand_dataset_per_task(combined["train"], SUBTASKS)
    combined["validation"] = expand_dataset_per_task(combined["test"], SUBTASKS)

    combined["test"] = expand_dataset_per_task(combined["test"], SUBTASKS)

    # get all the unique tags and add to feature information
    label_names = {
        task: ["O"] + [f"{pre}-{label}" for pre in ["B", "I"] for label in tags]
        for task, tags in zip(SUBTASKS, LABELS)
    }

    for split in ["train", "validation", "test"]:
        for st in SUBTASKS:
            combined[split][st].features["tags"] = datasets.Sequence(
                feature=datasets.ClassLabel(names=label_names[st])
            )

    return combined
