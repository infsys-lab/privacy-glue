#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from itertools import chain, repeat
from privacy_glue import main
import pytest
import os


def repeat_elements_n_times(input_list, n):
    return list(chain.from_iterable(repeat(element, n) for element in input_list))


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
def test_main(
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
    data_args, model_args, train_args = mocked_arguments(
        task=task,
        model_name_or_path=model_name_or_path,
        do_summarize=do_summarize,
        random_seed_iterations=random_seed_iterations,
        output_dir=output_dir,
        report_to=report_to,
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
    )

    summarize = mocker.patch("privacy_glue.summarize")
    seq_class = mocker.patch(
        "privacy_glue.Sequence_Classification_Pipeline", new_callable=deep_mocker
    )
    seq_tag = mocker.patch(
        "privacy_glue.Sequence_Tagging_Pipeline", new_callable=deep_mocker
    )
    read_comp = mocker.patch(
        "privacy_glue.Reading_Comprehension_Pipeline", new_callable=deep_mocker
    )

    # execute main
    main()

    # gather all calls made
    all_calls = (
        seq_class.call_args_list + seq_tag.call_args_list + read_comp.call_args_list
    )

    # check calls based on task
    if task in [
        "opp_115",
        "policy_detection",
        "policy_ie_a",
        "privacy_qa",
    ]:
        seq_class.assert_called()
        seq_tag.assert_not_called()
        read_comp.assert_not_called()
    elif task in ["piextract", "policy_ie_b"]:
        seq_class.assert_not_called()
        seq_tag.assert_called()
        read_comp.assert_not_called()
    elif task == "policy_qa":
        seq_class.assert_not_called()
        seq_tag.assert_not_called()
        read_comp.assert_called()
    elif task == "all":
        seq_class.assert_called()
        seq_tag.assert_called()
        read_comp.assert_called()

    # check that correct tasks were executed
    assert [call[0][0].task for call in all_calls] == repeat_elements_n_times(
        task_aggregated,
        random_seed_iterations,
    )

    # check that models were called correctly
    assert [
        call[0][1].model_name_or_path for call in all_calls
    ] == repeat_elements_n_times(
        [model_name_or_path], random_seed_iterations * len(task_aggregated)
    )

    # check that reporting was done correctly
    assert [call[0][2].report_to for call in all_calls] == repeat_elements_n_times(
        [report_to], random_seed_iterations * len(task_aggregated)
    )

    # check that summarize behaviour is correct
    if do_summarize:
        summarize.assert_called_once_with(os.path.join(output_dir, model_dir_basename))
    else:
        summarize.assert_not_called()
