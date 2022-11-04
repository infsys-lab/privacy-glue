#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from parser import get_parser

from utils.experiment_utils import Privacy_GLUE_Experiment_Manager


def main() -> None:
    # get parser and parse arguments
    parser = get_parser()
    (
        data_args,
        model_args,
        train_args,
        experiment_args,
    ) = parser.parse_args_into_dataclasses()

    # create and run experiment manager
    Privacy_GLUE_Experiment_Manager(
        data_args, model_args, train_args, experiment_args
    ).run_experiments()


if __name__ == "__main__":
    main()
