#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import shutil

import numpy as np
import pytest

from utils.experiment_utils import Privacy_GLUE_Experiment_Manager


@pytest.mark.parametrize(
    "model_name_or_path, model_dir_basename",
    [
        ("bert-base-uncased", "bert_base_uncased"),
        ("nlpaueb/legal-bert-base-uncased", "nlpaueb_legal_bert_base_uncased"),
    ],
)
def test__init__(model_name_or_path, model_dir_basename, mocked_arguments):
    data_args, model_args, train_args, experiment_args = mocked_arguments(
        model_name_or_path=model_name_or_path, with_experiment_args=True
    )
    experiment_manager = Privacy_GLUE_Experiment_Manager(
        data_args, model_args, train_args, experiment_args
    )
    assert experiment_manager.data_args == data_args
    assert experiment_manager.model_args == model_args
    assert experiment_manager.train_args == train_args
    assert experiment_manager.experiment_args == experiment_args
    assert experiment_manager.experiment_args.model_dir == os.path.join(
        train_args.output_dir, model_dir_basename
    )


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
    parser = mocker.patch("privacy_glue.get_parser")
    parser.return_value.parse_args_into_dataclasses.return_value = (
        data_args,
        model_args,
        train_args,
        experiment_args,
    )
    summarize = mocker.patch.object(experiment_manager, "summarize")
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
                wandb_group=model_name_or_path if report_to == ["wandb"] else None,
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


@pytest.mark.parametrize(
    "model_name_or_path, model_dir_basename",
    [
        ("bert-base-uncased", "bert_base_uncased"),
        ("nlpaueb/legal-bert-base-uncased", "nlpaueb_legal_bert_base_uncased"),
    ],
)
@pytest.mark.parametrize(
    "random_seed_iterations",
    [1, 5, 10],
)
@pytest.mark.parametrize("add_unrelated_directory", [True, False])
@pytest.mark.parametrize("missing_task", [True, False])
@pytest.mark.parametrize("missing_results_file", [True, False])
def test_summarize(
    model_name_or_path,
    model_dir_basename,
    random_seed_iterations,
    add_unrelated_directory,
    missing_task,
    missing_results_file,
    mocked_arguments_with_tmp_path,
    mocker,
):
    # get mocked arguments and create experiment manager
    data_args, model_args, train_args, experiment_args = mocked_arguments_with_tmp_path(
        model_name_or_path=model_name_or_path,
        random_seed_iterations=random_seed_iterations,
        with_experiment_args=True,
    )
    experiment_manager = Privacy_GLUE_Experiment_Manager(
        data_args, model_args, train_args, experiment_args
    )
    experiment_manager.experiment_args.model_dir = os.path.join(
        train_args.output_dir, model_dir_basename
    )

    # create mocks
    warn = mocker.patch("utils.experiment_utils.warnings.warn")

    # create expected summary to fill up
    expected_benchmark_summary = {}

    # set random seed before for-loop
    np.random.seed(42)

    # create directory structure and data
    for task, metric_names in experiment_manager.task_metrics.items():
        # skip task if we should test this
        if missing_task and task == "piextract":
            continue

        # create task directory
        task_dir = os.path.join(experiment_manager.experiment_args.model_dir, task)
        os.makedirs(task_dir)

        # generate random (seeded) metric values
        metric_by_seed_group = [
            [np.random.uniform() for _ in metric_names]
            for _ in range(random_seed_iterations)
        ]

        # simulate a missing file
        if missing_results_file:
            metric_by_seed_group = metric_by_seed_group[: (random_seed_iterations - 1)]

        # pre-compute mean and standard deviations for checks
        metric_by_group_seed = list(zip(*metric_by_seed_group))
        expected_benchmark_summary[task] = {
            "metrics": metric_names,
            "mean": [
                np.round(np.mean(metric_group), 8).item()
                for metric_group in metric_by_group_seed
            ],
            "std": [
                np.round(np.std(metric_group), 8).item()
                for metric_group in metric_by_group_seed
            ],
            "num_samples": len(metric_by_seed_group),
        }

        # iterate over random seeds
        for index, seed in enumerate(range(random_seed_iterations)):
            # create seed directory
            seed_dir = os.path.join(task_dir, f"seed_{seed}")
            os.makedirs(seed_dir)

            if missing_results_file and index == (random_seed_iterations - 1):
                continue
            else:
                # create metrics dictionary
                metrics_dump = {
                    f"predict_{metric_name}": metric_by_seed_group[index][sub_index]
                    for sub_index, metric_name in enumerate(metric_names)
                }

                # dump metrics file
                with open(
                    os.path.join(seed_dir, "all_results.json"), "w"
                ) as output_file_stream:
                    json.dump(metrics_dump, output_file_stream)

    # add unrelated directory if required
    if add_unrelated_directory:
        shutil.copytree(
            os.path.join(experiment_manager.experiment_args.model_dir, "policy_qa"),
            os.path.join(experiment_manager.experiment_args.model_dir, "misc"),
        )

    # execute manager method
    experiment_manager.summarize()

    # assert conditional warnings
    if missing_results_file:
        warn.assert_called()
    else:
        warn.assert_not_called()

    # read JSON file
    with open(
        os.path.join(
            experiment_manager.experiment_args.model_dir, "benchmark_summary.json"
        )
    ) as input_file_stream:
        benchmark_summary = json.load(input_file_stream)

    # make assertion
    assert benchmark_summary == expected_benchmark_summary
