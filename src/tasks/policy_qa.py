#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from glob import glob
import datasets
import json
import os


def load_policy_qa(directory: str) -> datasets.DatasetDict:
    # define DatasetDict for data storage
    combined = datasets.DatasetDict()

    # loop over JSON files
    for json_file in glob(os.path.join(directory, "*.json")):
        # infer split from filename
        filename = os.path.basename(json_file)
        split = "validation" if filename.startswith(
            "dev") else filename.replace(".json", "")

        # define temporarily dictionary
        temp_dict = {
            "id": [],
            "title": [],
            "context": [],
            "question": [],
            "answers": []
        }

        # read JSON file
        with open(json_file, "r") as input_file_stream:
            dataset = json.load(input_file_stream)

        # loop over data and save to dictionray
        for article in dataset["data"]:
            title = article["title"]
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]
                answers = {}
                for qa in paragraph["qas"]:
                    answers["text"] = [
                        answer["text"] for answer in qa["answers"]
                    ]
                    answers["answer_start"] = [
                        answer["answer_start"] for answer in qa["answers"]
                    ]
                    temp_dict["id"].append(qa["id"])
                    temp_dict["title"].append(title)
                    temp_dict["context"].append(context)
                    temp_dict["question"].append(qa["question"])
                    temp_dict["answers"].append(answers)

        # convert temp_dict to Dataset and insert into DatasetDict
        combined[split] = datasets.Dataset.from_dict(temp_dict)

    return combined
