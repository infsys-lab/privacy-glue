#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict
import datasets
import os


def load_opt_out(directory: str) -> Dict[str, datasets.DatasetDict]:
    binary_data = datasets.load_dataset("json",
                                        data_files=os.path.join(
                                            directory, "binary_data",
                                            "binary_data.json"),
                                        split="all")
    binary_data = binary_data.train_test_split(test_size=0.3, seed=42)
    category_data = datasets.load_dataset(
        "json",
        data_files={
            "train": os.path.join(directory, "category_data",
                                  "train_set.jsonl"),
            "test": os.path.join(directory, "category_data", "test_set.jsonl")
        })
    category_data = category_data.remove_columns("Policy Url")
    category_data = category_data.rename_columns({
        "Opt Out Url": "url",
        "Sentence Text": "full_sentence_text",
        "Hyperlink Text": "hyperlink_text",
        "Labels": "label"
    })
    return {"binary": binary_data, "category": category_data}
