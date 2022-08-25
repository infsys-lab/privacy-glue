#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict
from typing import Dict, Tuple, Iterable, Any
from tqdm import tqdm
from glob import glob
import datasets
import yaml
import nltk
import os
import re

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def has_conflicting_labels(labels: Iterable[Tuple[Any, Any]]) -> bool:
    # create defaultdict object
    label_mapping = defaultdict(list)

    # map labels to defaultdict
    for key, value in labels:
        label_mapping[key].append(value)

    # check if there are any cases with multiple entries
    for key, value in label_mapping.items():
        if len(value) > 1:
            return True

    return False


def get_normalized_label(label: Dict[str, str]) -> Tuple[str, str, str]:
    # conditionally segment label
    if label["practice"] not in ["SSO", "Facebook_SSO"]:
        split_label = tuple(
            ("%s %s" % (re.sub(r"^(.*)_((1st|3rd)Party$)", r"\1 \2",
                               label["practice"]), label["modality"])).split())
    else:
        split_label = (label["practice"], "1stParty", label["modality"])

    return split_label


def load_app_350(directory: str) -> datasets.DatasetDict:
    # create global documents defaultdict structure
    documents = defaultdict(list)

    # loop over all YAML files
    for file in tqdm(glob(os.path.join(directory, "annotations", "*.yml")),
                     desc="Parsing YAML files"):
        # read YAML files using a fast loader
        with open(file, "r") as input_file_stream:
            document = yaml.load(input_file_stream, Loader=Loader)

        # infer split based on metadata
        split = re.sub(r"ing$", "", document["policy_type"].lower())

        # loop over all segments in document
        for segment in document["segments"]:
            # check if the current segment has one or more difficult annotation
            # NOTE: a difficult annotation is where an annotation is equal to
            # the entire segment or if it is not included in the segment (which
            # could imply some text formatting issues)
            has_difficult_annotation = any([
                ann_sentence["sentence_text"] not in segment["segment_text"]
                or ann_sentence["sentence_text"] == segment["segment_text"]
                for ann_sentence in segment["sentences"]
            ])

            # proceed conditionally depending on the presence of a difficult
            # annotation in the segment
            if has_difficult_annotation:
                # loop over all sentences in segment
                for ann_sentence in segment["sentences"]:
                    # for each sentence, gather all labels and normalize them
                    labels = tuple(
                        set([
                            get_normalized_label(annotation)
                            for annotation in ann_sentence["annotations"]
                        ]))

                    # assert that the labels are not conflicting
                    assert not has_conflicting_labels(
                        [((label[0], label[1]), label[2]) for label in labels])

                    # check if the extracted sentence has already been seen
                    if ann_sentence["sentence_text"] in documents["text"]:
                        # if the sentence was seen, find its index
                        index = documents["text"].index(
                            ann_sentence["sentence_text"])

                        # check if the label(s) of the current sentence and
                        # the already seen sentence are the same
                        if documents["label"][index] == labels:
                            # if they are the same, move on from this sentence
                            continue
                        else:
                            # if they are not the same, concatenate the set
                            # of all labels to make them unique
                            documents["label"][index] = tuple(
                                set(documents["label"][index] + labels))

                            # check if the new labels are conflicting
                            if has_conflicting_labels([
                                ((label[0], label[1]), label[2])
                                    for label in documents["label"][index]
                            ]):
                                # if any conflict is detected, make the
                                # sentence a negative example
                                documents["label"][index] = ()
                    else:
                        # if the sentence was not seen, append it to documents
                        documents["text"].append(ann_sentence["sentence_text"])
                        documents["split"].append(split)
                        documents["label"].append(labels)
            else:
                # since no difficult annotations are present in the segment,
                # we break the segment into sentences using NLTK
                sentences = nltk.sent_tokenize(segment["segment_text"])

                # loop over all NLTK sentences detected
                for sentence in sentences:
                    # create a sentence mapping list
                    sentence_mapping = []

                    # loop over all annotated sentences
                    # NOTE: this will be skipped if there is no annotation
                    for index, ann_sentence in enumerate(segment["sentences"]):
                        if ann_sentence["sentence_text"] in sentence:
                            # if an annotated sentence occurs within
                            # a NLTK sentence, record its index for
                            # subsequent aggregation
                            sentence_mapping.append(index)

                    # check if there is any sentence mapping
                    if sentence_mapping:
                        # if there is a sentence mapping, gather all
                        # labels from annotated sentences to associate
                        # with the NLTK sentence
                        labels = tuple(
                            set([
                                get_normalized_label(annotation)
                                for index in sentence_mapping for annotation in
                                segment["sentences"][index]["annotations"]
                            ]))

                        # ensure there are no conflicting labels
                        assert not has_conflicting_labels(
                            [((label[0], label[1]), label[2])
                             for label in labels])
                    else:
                        # if there is no sentence mapping, then declare this
                        # NLTK sentence as a negative example
                        labels = ()

                    # check if the extracted sentence has already been seen
                    if sentence in documents["text"]:
                        # if the sentence was seen, find its index
                        index = documents["text"].index(sentence)

                        # check if the label(s) of the current sentence and
                        # the already seen sentence are the same
                        if documents["label"][index] == labels:
                            # if they are the same, move on from this sentence
                            continue
                        else:
                            # if they are not the same, concatenate the set
                            # of all labels to make them unique
                            documents["label"][index] = tuple(
                                set(documents["label"][index] + labels))

                            # check if the new labels are conflicting
                            if has_conflicting_labels([
                                ((label[0], label[1]), label[2])
                                    for label in documents["label"][index]
                            ]):
                                # if any conflict is detected, make the
                                # sentence a negative example
                                documents["label"][index] = ()
                    else:
                        # if the sentence was not seen, append it to documents
                        documents["text"].append(sentence)
                        documents["split"].append(split)
                        documents["label"].append(labels)

    # ensure that all text-label pairs are unique
    unique_pairs = set(zip(documents["text"], documents["label"]))
    assert len(unique_pairs) == len(documents["text"]) == len(
        documents["label"])

    # ensure that there are no global
    assert not has_conflicting_labels(unique_pairs)

    # replace all empty tuples with negative label
    for index, label in enumerate(documents["label"]):
        if not label:
            documents["label"][index] = (("Negative", ), )

    # convert documents into HF dataset
    dataset = datasets.Dataset.from_dict(documents)

    # create a DatasetDict to fill up
    combined = datasets.DatasetDict()

    # assign splits to DatasetDict and drop the "split" column
    for split in ["train", "validation", "test"]:
        combined[split] = dataset.filter(
            lambda row: row["split"] == split).remove_columns("split")

    return combined
