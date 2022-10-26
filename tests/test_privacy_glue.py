#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from privacy_glue import main


def test_main(mocker):
    # mock relevant objects
    parser = mocker.patch("privacy_glue.get_parser")
    parser.return_value.parse_args_into_dataclasses.return_value = (
        "data_args",
        "model_args",
        "train_args",
        "experiment_args",
    )
    experiment_manager = mocker.patch("privacy_glue.Privacy_GLUE_Experiment_Manager")

    # execute function
    main()

    # make assertions
    assert parser.mock_calls == [
        mocker.call(),
        mocker.call().parse_args_into_dataclasses(),
    ]
    assert experiment_manager.mock_calls == [
        mocker.call("data_args", "model_args", "train_args", "experiment_args"),
        mocker.call().run_experiments(),
    ]
