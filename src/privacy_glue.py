#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from parser import MODELS, TASKS, get_parser
from sequence_classification import Sequence_Classification_Pipeline
import multiprocessing as mp
import wandb
import copy
import os
import re

GPUS = os.environ["CUDA_VISIBLE_DEVICES"].split(",")


def summarize(model_dir: str) -> None:
    pass
    # raise NotImplementedError


def start_wandb(task, seed):
    # initialize weights and biases
    # NOTE: There are several approaches to managing multiprocess training:
    #   1. Call wandb.init in all your processes, using the group keyword
    #      argument to define a shared group. Each process will have its own
    #      wandb run and the UI will group the training processes together.
    #   2. Call wandb.init from just one process and pass data to be logged
    #      over multiprocessing queues.
    return wandb.init(
        name=f'{os.environ["WANDB_RUN_GROUP"][11:]}\
            _seed_{str(seed)}',
        project="privacyGLUE-" + task,
        entity="profila",
        reinit=True,
    )


class Benchmark_seed:
    def __init__(self, seed, data_args, model_args, train_args, model_dir):
        self.seed = seed
        self.data_args = data_args
        self.model_args = model_args
        self.train_args = train_args
        self.model_dir = model_dir


# TODO: add scipy explicitly to deps
# TODO: rename seed_object to some thing more meaningful
def run_benchmark_for_seed(payload):
    try:
        seed_object, queue = payload
        ident = mp.current_process().ident
        gpu_id = queue.get()
        seed_object.train_args.gpu_id = gpu_id
        print(
            "Process {}: task {} with model {} and seed {} on GPU {}".format(
                ident,
                seed_object.data_args.task,
                seed_object.model_args.model_name_or_path,
                seed_object.seed,
                gpu_id,
            )
        )
        # wandb_run = start_wandb(seed_object.data_args.task, seed_object.seed)
        # branch into separate workflows depending on task type
        if seed_object.data_args.task in [
            "opp_115",
            "policy_detection",
            "policy_ie_a",
            "privacy_qa",
        ]:
            pl = Sequence_Classification_Pipeline(
                seed_object.data_args,
                seed_object.model_args,
                seed_object.train_args,
            )
            pl.run_pipeline()
        elif seed_object.data_args.task in ["piextract", "policy_ie_b"]:
            raise NotImplementedError
        elif seed_object.data_args.task == "policy_qa":
            raise NotImplementedError
    finally:
        # wandb_run.finish()
        queue.put(gpu_id)


def main() -> None:
    # get parser and parse arguments
    parser = get_parser()
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()

    # capture base output directory
    output_dir = train_args.output_dir

    # decide iteration strategy
    if data_args.task != "all":
        tasks = [data_args.task]
    else:
        tasks = [task for task in TASKS if task != "all"]

    # decide iteration strategy
    if model_args.model_name_or_path != "all":
        models = [model_args.model_name_or_path]
    else:
        models = [model for model in MODELS if model != "all"]

    # find out how many GPUS are available
    num_gpus = len(GPUS)
    # Add the GPU ids to the queue
    for gpu_id in range(num_gpus):
        queue.put(gpu_id)

    # TODO: conditionally decide batch size
    # replicate seed objects for the processes
    seeds = []
    for model in models:
        model_args.model_name_or_path = model
        model_dir = os.path.join(
            output_dir,
            re.sub(r"[/-]", "_", model_args.model_name_or_path),
        )
        for task in tasks:
            data_args.task = task
            for seed_id in range(
                model_args.random_seed_iterations
            ):  # for seed ins seeds for model in models
                train_args.output_dir = os.path.join(
                    model_dir,
                    re.sub(r"[/-]", "_", data_args.task),
                    "seed_%s" % seed_id,
                )
                seeds.append(
                    (
                        Benchmark_seed(
                            seed_id,
                            copy.deepcopy(data_args),
                            copy.deepcopy(model_args),
                            copy.deepcopy(train_args),
                            model_dir,
                        ),
                        queue,
                    )
                )

    # os.environ["WANDB_RUN_GROUP"] = f"experiment_{wandb.util.generate_id()}"
    # run <number of available GPU> processes in parallel and distribute the seed runs
    with context.Pool(processes=num_gpus) as pool:
        for _ in pool.imap_unordered(run_benchmark_for_seed, seeds):
            pass

    # summarize PrivacyGLUE benchmark
    if model_args.do_summarize:
        summarize(model_dir)


if __name__ == "__main__":
    context = mp.get_context("spawn")
    m = context.Manager()
    queue = m.Queue()
    main()
