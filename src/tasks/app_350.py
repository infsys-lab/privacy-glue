#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict
from typing import Dict, Tuple, Iterable, Any
from glob import glob
import datasets
import json
import nltk
import os
import re


def has_conflicting_labels(labels: Iterable[Tuple[Any, Any]]) -> bool:
    label_mapping = defaultdict(list)
    for key, value in labels:
        label_mapping[key].append(value)
    for key, value in label_mapping.items():
        if len(value) > 1:
            return True
    return False


def get_normalized_label(label: Dict[str, str]) -> Tuple[str, str, str]:
    if label["practice"] not in ["SSO", "Facebook_SSO"]:
        split_label = tuple(
            ("%s %s" % (re.sub(r"^(.*)_((1st|3rd)Party$)", r"\1 \2",
                               label["practice"]), label["modality"])).split())
    else:
        split_label = (label["practice"], "1stParty", label["modality"])
    return split_label


def load_app_350(directory: str) -> datasets.DatasetDict:
    documents = defaultdict(list)
    for file in glob(os.path.join(directory, "annotations", "*.json")):
        with open(file, "r") as input_file_stream:
            document = json.load(input_file_stream)
        split = re.sub(r"ing$", "", document["policy_type"].lower())
        for segment in document["segments"]:
            has_difficult_annotation = any([
                ann_sentence["sentence_text"] not in segment["segment_text"]
                or ann_sentence["sentence_text"] == segment["segment_text"]
                for ann_sentence in segment["sentences"]
            ])

            if has_difficult_annotation:
                for ann_sentence in segment["sentences"]:
                    labels = tuple(
                        set([
                            get_normalized_label(annotation)
                            for annotation in ann_sentence["annotations"]
                        ]))

                    assert not has_conflicting_labels(
                        [((label[0], label[1]), label[2]) for label in labels])

                    if ann_sentence["sentence_text"] in documents["text"]:
                        index = documents["text"].index(
                            ann_sentence["sentence_text"])
                        if documents["label"][index] == labels:
                            continue
                        else:
                            documents["label"][index] = tuple(
                                set(documents["label"][index] + labels))
                            if has_conflicting_labels([
                                ((label[0], label[1]), label[2])
                                    for label in documents["label"][index]
                            ]):
                                documents["label"][index] = ()
                    else:
                        documents["text"].append(ann_sentence["sentence_text"])
                        documents["split"].append(split)
                        documents["label"].append(labels)
            else:
                sentences = nltk.sent_tokenize(segment["segment_text"])
                for sentence in sentences:
                    sentence_mapping = []
                    for index, ann_sentence in enumerate(segment["sentences"]):
                        if ann_sentence["sentence_text"] in sentence:
                            sentence_mapping.append(index)

                    if sentence_mapping:
                        labels = tuple(
                            set([
                                get_normalized_label(annotation)
                                for index in sentence_mapping for annotation in
                                segment["sentences"][index]["annotations"]
                            ]))
                        assert not has_conflicting_labels(
                            [((label[0], label[1]), label[2])
                             for label in labels])
                    else:
                        labels = ()

                    if sentence in documents["text"]:
                        index = documents["text"].index(sentence)
                        if documents["label"][index] == labels:
                            continue
                        else:
                            documents["label"][index] = tuple(
                                set(documents["label"][index] + labels))
                            if has_conflicting_labels([
                                ((label[0], label[1]), label[2])
                                    for label in documents["label"][index]
                            ]):
                                documents["label"][index] = ()
                    else:
                        documents["text"].append(sentence)
                        documents["split"].append(split)
                        documents["label"].append(labels)

    unique_pairs = set(zip(documents["text"], documents["label"]))
    assert len(unique_pairs) == len(documents["text"]) == len(
        documents["label"])
    assert not has_conflicting_labels(unique_pairs)

    dataset = datasets.Dataset.from_dict(documents)
    combined = datasets.DatasetDict()
    for split in ["train", "validation", "test"]:
        combined[split] = dataset.filter(
            lambda row: row["split"] == split).remove_columns("split")

    return combined
