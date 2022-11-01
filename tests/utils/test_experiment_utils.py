#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import pytest

from utils.experiment_utils import Privacy_GLUE_Experiment_Manager


def test__init__():
    experiment_manager = Privacy_GLUE_Experiment_Manager(
        "data_args", "model_args", "train_args", "experiment_args"
    )
    assert experiment_manager.data_args == "data_args"
    assert experiment_manager.model_args == "model_args"
    assert experiment_manager.train_args == "train_args"
    assert experiment_manager.experiment_args == "experiment_args"


@pytest.mark.parametrize(
    "task",
    [
        "all",
        "opp_115",
        "policy_detection",
        "policy_ie_a",
        "privacy_qa",
        "piextract",
        "policy_ie_b",
        "policy_qa",
    ],
)
@pytest.mark.parametrize(
    "model_name_or_path, model_dir_basename",
    [("bert-base-uncased", "bert_base_uncased")],
)
@pytest.mark.parametrize(
    "do_summarize",
    [True, False],
)
@pytest.mark.parametrize(
    "random_seed_iterations",
    [1, 5],
)
@pytest.mark.parametrize(
    "output_dir",
    ["/tmp/runs", "/tmp/another/runs"],
)
@pytest.mark.parametrize(
    "report_to",
    [[], ["wandb"]],
)
def test_run_experiments(
    task,
    model_name_or_path,
    do_summarize,
    random_seed_iterations,
    output_dir,
    report_to,
    model_dir_basename,
    mocked_arguments,
    mocker,
    deep_mocker,
):
    # get mocked arguments
    data_args, model_args, train_args, experiment_args = mocked_arguments(
        task=task,
        model_name_or_path=model_name_or_path,
        do_summarize=do_summarize,
        random_seed_iterations=random_seed_iterations,
        output_dir=output_dir,
        report_to=report_to,
        with_experiment_args=True,
    )
    experiment_manager = Privacy_GLUE_Experiment_Manager(
        data_args, model_args, train_args, experiment_args
    )

    # aggregate task for next steps.
    task_aggregated = (
        [task]
        if task != "all"
        else [
            "opp_115",
            "policy_detection",
            "policy_ie_a",
            "privacy_qa",
            "piextract",
            "policy_ie_b",
            "policy_qa",
        ]
    )

    # mock relevant modules/objects
    mocker.patch("utils.experiment_utils.generate_id", return_value="test")
    parser = mocker.patch("privacy_glue.get_parser")
    parser.return_value.parse_args_into_dataclasses.return_value = (
        data_args,
        model_args,
        train_args,
        experiment_args,
    )
    summarize = mocker.patch.object(experiment_manager, "_summarize")
    seq_class = mocker.patch(
        "utils.experiment_utils.Sequence_Classification_Pipeline",
        new_callable=deep_mocker,
    )
    seq_tag = mocker.patch(
        "utils.experiment_utils.Sequence_Tagging_Pipeline", new_callable=deep_mocker
    )
    read_comp = mocker.patch(
        "utils.experiment_utils.Reading_Comprehension_Pipeline",
        new_callable=deep_mocker,
    )

    # create expected calls
    expected_seq_class_calls = []
    expected_seq_tag_calls = []
    expected_read_comp_calls = []
    for _task in task_aggregated:
        for _seed in range(random_seed_iterations):
            data_args, model_args, train_args = mocked_arguments(
                seed=_seed,
                task=_task,
                model_name_or_path=model_name_or_path,
                output_dir=os.path.join(
                    output_dir, model_dir_basename, _task, f"seed_{_seed}"
                ),
                report_to=report_to,
                wandb_group_id="experiment_test" if report_to == ["wandb"] else None,
            )
            train_args.get_process_log_level = mocker.ANY
            train_args.main_process_first = mocker.ANY
            if _task in [
                "opp_115",
                "policy_detection",
                "policy_ie_a",
                "privacy_qa",
            ]:
                expected_seq_class_calls.extend(
                    [
                        mocker.call(data_args, model_args, train_args),
                        mocker.call().run_pipeline(),
                    ]
                )
            elif _task in ["piextract", "policy_ie_b"]:
                expected_seq_tag_calls.extend(
                    [
                        mocker.call(data_args, model_args, train_args),
                        mocker.call().run_pipeline(),
                    ]
                )
            elif _task == "policy_qa":
                expected_read_comp_calls.extend(
                    [
                        mocker.call(data_args, model_args, train_args),
                        mocker.call().run_pipeline(),
                    ]
                )

    # execute main
    experiment_manager.run_experiments()

    # make initial assertion on model directory
    assert experiment_manager.experiment_args.model_dir == os.path.join(
        output_dir, model_dir_basename
    )

    # make assertions on arguments based on task
    if task in [
        "opp_115",
        "policy_detection",
        "policy_ie_a",
        "privacy_qa",
    ]:
        assert seq_class.mock_calls == expected_seq_class_calls
        seq_tag.assert_not_called()
        read_comp.assert_not_called()
    elif task in ["piextract", "policy_ie_b"]:
        seq_class.assert_not_called()
        assert seq_tag.mock_calls == expected_seq_tag_calls
        read_comp.assert_not_called()
    elif task == "policy_qa":
        seq_class.assert_not_called()
        seq_tag.assert_not_called()
        assert read_comp.mock_calls == expected_read_comp_calls
    elif task == "all":
        assert seq_class.mock_calls == expected_seq_class_calls
        assert seq_tag.mock_calls == expected_seq_tag_calls
        assert read_comp.mock_calls == expected_read_comp_calls

    # check that summarize behaviour is correct
    if do_summarize:
        summarize.assert_called_once()
    else:
        summarize.assert_not_called()
