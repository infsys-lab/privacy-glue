#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datasets import DatasetDict
from parser import DataArguments
from tasks.opp_115 import load_opp_115
from tasks.piextract import load_piextract
from tasks.policy_detection import load_policy_detection
from tasks.policy_ie_a import load_policy_ie_a
from tasks.policy_ie_b import load_policy_ie_b
from tasks.policy_qa import load_policy_qa
from tasks.privacy_qa import load_privacy_qa
import os


def get_data(data_args: DataArguments) -> DatasetDict:
    # define new argument based on task name
    task_dir = os.path.join(data_args.data_dir, data_args.task)

    # load dataset based on task name
    if data_args.task == "opp_115":
        data = load_opp_115(task_dir)
    elif data_args.task == "piextract":
        data = load_piextract(task_dir)
    elif data_args.task == "policy_detection":
        data = load_policy_detection(task_dir)
    elif data_args.task == "policy_ie_a":
        data = load_policy_ie_a(task_dir)
    elif data_args.task == "policy_ie_b":
        data = load_policy_ie_b(task_dir)
    elif data_args.task == "policy_qa":
        data = load_policy_qa(task_dir)
    elif data_args.task == "privacy_qa":
        data = load_privacy_qa(task_dir)

    return data
