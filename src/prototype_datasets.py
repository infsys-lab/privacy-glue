#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict
from typing import Dict, Tuple, Iterable, Any
from datasets import load_dataset, concatenate_datasets
from glob import glob
import pandas as pd
import datasets
import json
import ipdb
import os
import re


def load_policy_detection(directory: str) -> datasets.DatasetDict:
    df = pd.read_csv(os.path.join(directory, "1301_dataset.csv"), index_col=0)
    df = df[["policy_text", "is_policy"]]
    df["is_policy"] = df["is_policy"].astype(int)
    df = df.rename(columns={"policy_text": "text", "is_policy": "label"})
    dataset = datasets.Dataset.from_pandas(df, preserve_index=False)
    return dataset.train_test_split(test_size=0.3, seed=42)


def load_privacy_qa(directory: str) -> datasets.DatasetDict:
    # load and process the train dataset
    train_df = pd.read_csv(os.path.join(directory, "policy_train.tsv"),
                           sep="\t")
    train_df = train_df[["DocID", "QueryID", "Query", "Segment", "Label"]]
    train_df["Label"] = train_df["Label"].apply(
        lambda x: x == "Relevant").astype(int)
    train_df = train_df.groupby(["DocID", "QueryID"]).agg({
        "Query":
        lambda x: x.unique(),
        "Segment":
        lambda x: x.tolist(),
        "Label":
        lambda x: x.tolist()
    })
    train_df = train_df.reset_index()[["QueryID", "Query", "Segment", "Label"]]
    train_df = train_df.rename(columns={
        "Query": "question",
        "Segment": "text",
        "Label": "label"
    })

    # load and process the train meta dataset
    train_meta_df = pd.read_csv(os.path.join(directory,
                                             "train_opp_annotations.tsv"),
                                sep="\t")
    train_meta_df = train_meta_df.drop(["Folder", "DocID", "Split"], axis=1)
    train_meta_df["opp_category"] = train_meta_df[[
        "first", "third", "datasecurity", "dataretention", "user_access",
        "user_choice", "other"
    ]].apply(lambda row: row[row == 1].index.tolist(), axis=1)
    assert set(train_df["QueryID"].tolist()) == set(
        train_meta_df["QueryID"].tolist())
    train_df = train_df.merge(train_meta_df[["QueryID", "opp_category"]],
                              on="QueryID")
    train_dataset = datasets.Dataset.from_pandas(train_df.drop("QueryID",
                                                               axis=1),
                                                 preserve_index=False)
    for sample in train_dataset:
        assert len(sample["text"]) == len(sample["label"])

    # work on the test dataset
    test_df = pd.read_csv(os.path.join(directory, "policy_test.tsv"), sep="\t")
    test_df = test_df[["DocID", "QueryID", "Query", "Segment", "Any_Relevant"]]
    test_df = test_df.rename(columns={"Any_Relevant": "Label"})
    test_df["Label"] = test_df["Label"].apply(
        lambda x: x == "Relevant").astype(int)
    test_df = test_df.groupby(["DocID", "QueryID"]).agg({
        "Query":
        lambda x: x.unique(),
        "Segment":
        lambda x: x.tolist(),
        "Label":
        lambda x: x.tolist()
    })
    test_df = test_df.reset_index()[["Query", "QueryID", "Segment", "Label"]]
    test_df = test_df.rename(columns={
        "Query": "question",
        "Segment": "text",
        "Label": "label"
    })

    # load and process the test meta dataset
    test_meta_df = pd.read_csv(os.path.join(directory,
                                            "test_opp_annotations.tsv"),
                               sep="\t")
    test_meta_df = test_meta_df.drop(["Folder", "DocID", "Split"], axis=1)
    test_meta_df["opp_category"] = test_meta_df[[
        "first", "third", "datasecurity", "dataretention", "user_access",
        "user_choice", "other"
    ]].apply(lambda row: row[row == 1].index.tolist(), axis=1)
    assert set(test_df["QueryID"].tolist()) == set(
        test_meta_df["QueryID"].tolist())
    test_df = test_df.merge(test_meta_df[["QueryID", "opp_category"]],
                            on="QueryID")
    test_dataset = datasets.Dataset.from_pandas(test_df.drop("QueryID",
                                                             axis=1),
                                                preserve_index=False)
    for sample in test_dataset:
        assert len(sample["text"]) == len(sample["label"])

    # return a combination of train and test datasets
    return datasets.DatasetDict({"train": train_dataset, "test": test_dataset})


def load_policy_qa(directory: str) -> datasets.DatasetDict:
    data_files = {}
    data_files["train"] = os.path.join(directory, "train.jsonl")
    data_files["validation"] = os.path.join(directory, "dev.jsonl")
    data_files["test"] = os.path.join(directory, "test.jsonl")
    return load_dataset("json", data_files=data_files).map(
        lambda example:
        {"question_type": example["question_type"].split("|||")})


def load_policy_ie(directory: str) -> datasets.DatasetDict:
    data_files = {}
    data_files["train"] = os.path.join(directory, "train", "seq.in")
    data_files["validation"] = os.path.join(directory, "valid", "seq.in")
    data_files["test"] = os.path.join(directory, "test", "seq.in")
    tokens = load_dataset("text", data_files=data_files).map(
        lambda example: {"tokens": example["text"].split()},
        remove_columns=["text"])
    data_files["train"] = os.path.join(directory, "train", "label")
    data_files["validation"] = os.path.join(directory, "valid", "label")
    data_files["test"] = os.path.join(directory, "test", "label")
    labels = load_dataset("text", data_files=data_files).rename_column(
        "text", "label")
    data_files["train"] = os.path.join(directory, "train", "seq_type_I.out")
    data_files["validation"] = os.path.join(directory, "valid",
                                            "seq_type_I.out")
    data_files["test"] = os.path.join(directory, "test", "seq_type_I.out")
    ner_tags_first = load_dataset("text", data_files=data_files).map(
        lambda example: {"ner_tags_type_one": example["text"].split()},
        remove_columns=["text"])
    data_files["train"] = os.path.join(directory, "train", "seq_type_II.out")
    data_files["validation"] = os.path.join(directory, "valid",
                                            "seq_type_II.out")
    data_files["test"] = os.path.join(directory, "test", "seq_type_II.out")
    ner_tags_second = load_dataset("text", data_files=data_files).map(
        lambda example: {"ner_tags_type_two": example["text"].split()},
        remove_columns=["text"])
    combined = datasets.DatasetDict()
    for (split, a), (_, b), (_, c), (_, d) in zip(tokens.items(),
                                                  labels.items(),
                                                  ner_tags_first.items(),
                                                  ner_tags_second.items()):
        combined[split] = concatenate_datasets([a, b, c, d], axis=1)
    return combined


def load_opt_out(directory: str) -> Dict[str, datasets.DatasetDict]:
    binary_data = load_dataset("json",
                               data_files=os.path.join(directory,
                                                       "binary_data",
                                                       "binary_data.json"),
                               split="all")
    binary_data = binary_data.train_test_split(test_size=0.3, seed=42)
    category_data = load_dataset("json",
                                 data_files={
                                     "train":
                                     os.path.join(directory, "category_data",
                                                  "train_set.jsonl"),
                                     "test":
                                     os.path.join(directory, "category_data",
                                                  "test_set.jsonl")
                                 })
    category_data = category_data.remove_columns("Policy Url")
    category_data = category_data.rename_columns({
        "Opt Out Url": "url",
        "Sentence Text": "full_sentence_text",
        "Hyperlink Text": "hyperlink_text",
        "Labels": "label"
    })
    return {"binary": binary_data, "category": category_data}


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


def load_opp_115(directory: str) -> datasets.DatasetDict:
    columns = ["text", "label"]
    df = pd.DataFrame()
    for split in ["train", "validation", "test"]:
        temp_df = pd.read_csv(os.path.join(directory,
                                           "%s_dataset.csv" % split),
                              header=None,
                              names=columns)
        temp_df = temp_df.groupby("text").agg(
            dict(label=lambda x: list(set(x)))).reset_index()
        temp_df["split"] = split
        df = pd.concat([df, temp_df], ignore_index=True)

    dataset = datasets.Dataset.from_pandas(df, preserve_index=False)
    combined = datasets.DatasetDict()
    for split in ["train", "validation", "test"]:
        combined[split] = dataset.filter(
            lambda row: row["split"] == split).remove_columns("split")
    return combined


# data = load_policy_detection("./data/policy_detection")
# data = load_privacy_qa("./data/privacy_qa")
# data = load_policy_qa("./data/policy_qa")
# data = load_policy_ie("./data/policy_ie")
# data = load_opt_out("./data/opt_out")
# data = load_app_350("./data/app_350")
data = load_opp_115("./data/opp_115")
