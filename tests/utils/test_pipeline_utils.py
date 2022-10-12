#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils.pipeline_utils import Privacy_GLUE_Pipeline, SuccessFileFoundException
import transformers
import datasets
import logging
import pytest
import os


class Mocked_Pipeline(Privacy_GLUE_Pipeline):
    def _retrieve_data(self):
        pass

    def _load_pretrained_model_and_tokenizer(self):
        pass

    def _apply_preprocessing(self):
        pass

    def _set_metrics(self):
        pass

    def _run_train_loop(self):
        pass


@pytest.mark.parametrize(
    "task",
    [
        "opp_115",
        "piextract",
        "policy_detection",
        "policy_ie_a",
        "policy_ie_b",
        "policy_qa",
        "privacy_qa",
    ],
)
def test__init__(task, mocked_arguments):
    mocked_pipeline = Mocked_Pipeline(*mocked_arguments(task=task))
    assert mocked_pipeline.data_args.task == task
    assert mocked_pipeline.model_args.model_name_or_path == "bert-base-uncased"
    assert mocked_pipeline.train_args.report_to == []
    assert mocked_pipeline.success_file == ".success"


@pytest.mark.parametrize(
    "task",
    [
        "opp_115",
        "piextract",
        "policy_detection",
        "policy_ie_a",
        "policy_ie_b",
        "policy_qa",
        "privacy_qa",
    ],
)
def test__get_data(task, mocked_arguments, mocker):
    # create mocked pipeline class
    mocked_pipeline = Mocked_Pipeline(*mocked_arguments(task=task))

    # mock leaf functions
    opp_115 = mocker.patch("utils.pipeline_utils.load_opp_115")
    piextract = mocker.patch("utils.pipeline_utils.load_piextract")
    policy_detection = mocker.patch("utils.pipeline_utils.load_policy_detection")
    policy_ie_a = mocker.patch("utils.pipeline_utils.load_policy_ie_a")
    policy_ie_b = mocker.patch("utils.pipeline_utils.load_policy_ie_b")
    policy_qa = mocker.patch("utils.pipeline_utils.load_policy_qa")
    privacy_qa = mocker.patch("utils.pipeline_utils.load_privacy_qa")

    # execute _get_data
    mocked_pipeline._get_data()

    # check exeuctions
    if task == "opp_115":
        opp_115.assert_called_once()
        piextract.assert_not_called()
        policy_detection.assert_not_called()
        policy_ie_a.assert_not_called()
        policy_ie_b.assert_not_called()
        policy_qa.assert_not_called()
        privacy_qa.assert_not_called()
    elif task == "piextract":
        opp_115.assert_not_called()
        piextract.assert_called_once()
        policy_detection.assert_not_called()
        policy_ie_a.assert_not_called()
        policy_ie_b.assert_not_called()
        policy_qa.assert_not_called()
        privacy_qa.assert_not_called()
    elif task == "policy_detection":
        opp_115.assert_not_called()
        piextract.assert_not_called()
        policy_detection.assert_called_once()
        policy_ie_a.assert_not_called()
        policy_ie_b.assert_not_called()
        policy_qa.assert_not_called()
        privacy_qa.assert_not_called()
    elif task == "policy_ie_a":
        opp_115.assert_not_called()
        piextract.assert_not_called()
        policy_detection.assert_not_called()
        policy_ie_a.assert_called_once()
        policy_ie_b.assert_not_called()
        policy_qa.assert_not_called()
        privacy_qa.assert_not_called()
    elif task == "policy_ie_b":
        opp_115.assert_not_called()
        piextract.assert_not_called()
        policy_detection.assert_not_called()
        policy_ie_a.assert_not_called()
        policy_ie_b.assert_called_once()
        policy_qa.assert_not_called()
        privacy_qa.assert_not_called()
    elif task == "policy_qa":
        opp_115.assert_not_called()
        piextract.assert_not_called()
        policy_detection.assert_not_called()
        policy_ie_a.assert_not_called()
        policy_ie_b.assert_not_called()
        policy_qa.assert_called_once()
        privacy_qa.assert_not_called()
    elif task == "privacy_qa":
        opp_115.assert_not_called()
        piextract.assert_not_called()
        policy_detection.assert_not_called()
        policy_ie_a.assert_not_called()
        policy_ie_b.assert_not_called()
        policy_qa.assert_not_called()
        privacy_qa.assert_called_once()


@pytest.mark.parametrize(
    "overwrite_output_dir",
    [True, False],
)
def test__init_run_dir(overwrite_output_dir, mocked_arguments_with_tmp_path, mocker):
    # create mocked pipeline class
    mocked_pipeline_with_tmp_path = Mocked_Pipeline(
        *mocked_arguments_with_tmp_path(overwrite_output_dir=overwrite_output_dir)
    )

    # add mocked functions
    shutil_rmtree = mocker.patch("utils.pipeline_utils.shutil.rmtree")
    os_makedirs = mocker.patch("utils.pipeline_utils.os.makedirs")

    # call __init_run_dir method
    mocked_pipeline_with_tmp_path._init_run_dir()

    # check effects of this
    if overwrite_output_dir:
        shutil_rmtree.assert_called_once()
        os_makedirs.assert_called_once()
    else:
        shutil_rmtree.assert_not_called()
        os_makedirs.assert_called_once()


@pytest.mark.parametrize(
    "log_level",
    ["info", "warning"],
)
def test__init_root_logger(log_level, mocked_arguments_with_tmp_path, mocker):
    # create mocked pipeline class
    mocked_pipeline_with_tmp_path = Mocked_Pipeline(
        *mocked_arguments_with_tmp_path(log_level=log_level)
    )

    # create mock objects
    init_logger = mocker.patch("utils.pipeline_utils.init_logger")
    add_file_handler = mocker.patch("utils.pipeline_utils.add_file_handler")

    # perform action
    mocked_pipeline_with_tmp_path._init_root_logger()

    # perform assertions
    assert hasattr(mocked_pipeline_with_tmp_path, "logger")
    init_logger.assert_called_once_with(
        mocked_pipeline_with_tmp_path.logger, logging.getLevelName(log_level.upper())
    )
    add_file_handler.assert_called_once_with(
        mocked_pipeline_with_tmp_path.logger,
        logging.getLevelName(log_level.upper()),
        os.path.join(
            mocked_pipeline_with_tmp_path.train_args.output_dir, "session.log"
        ),
    )


@pytest.mark.parametrize(
    "log_level",
    ["info", "warning"],
)
def test__init_third_party_loggers(log_level, mocked_arguments, mocker):
    # create mocked pipeline class
    mocked_pipeline = Mocked_Pipeline(*mocked_arguments(log_level=log_level))

    # execute method
    mocked_pipeline._init_third_party_loggers()

    # define loggers
    transformers_logger = transformers.utils.logging.get_logger()
    datasets_logger = datasets.utils.logging.get_logger()

    # make assertions about logging level
    assert transformers_logger.getEffectiveLevel() == logging.getLevelName(
        log_level.upper()
    )
    assert datasets_logger.getEffectiveLevel() == logging.getLevelName(
        log_level.upper()
    )

    # make assertions about handlers
    assert transformers_logger.handlers == []
    assert datasets_logger.handlers == []

    # make assertions about propagation
    assert transformers_logger.propagate
    assert datasets_logger.propagate


@pytest.mark.parametrize(
    "overwrite_output_dir",
    [True, False],
)
@pytest.mark.parametrize(
    "do_train",
    [True, False],
)
@pytest.mark.parametrize(
    "create_file",
    [True, False],
)
def test__check_for_success_file(
    overwrite_output_dir, do_train, create_file, mocked_arguments_with_tmp_path, mocker
):
    # create mocked pipeline class
    mocked_pipeline_with_tmp_path = Mocked_Pipeline(
        *mocked_arguments_with_tmp_path(
            **{"overwrite_output_dir": overwrite_output_dir, "do_train": do_train}
        )
    )

    # mock logger in pipeline
    mocker.patch.object(mocked_pipeline_with_tmp_path, "logger", create=True)

    # proceed with scenarios conditionally
    if create_file:
        with open(
            os.path.join(
                mocked_pipeline_with_tmp_path.train_args.output_dir, ".success"
            ),
            "w",
        ):
            if do_train and not overwrite_output_dir:
                with pytest.raises(SuccessFileFoundException):
                    mocked_pipeline_with_tmp_path._check_for_success_file()
            else:
                mocked_pipeline_with_tmp_path._check_for_success_file()
    else:
        mocked_pipeline_with_tmp_path._check_for_success_file()


def test__dump_misc_args(mocked_arguments, mocker):
    # create mocked pipeline class
    mocked_pipeline = Mocked_Pipeline(*mocked_arguments())

    # mock torch save
    torch_save = mocker.patch("utils.pipeline_utils.torch.save")

    # call the pipeline method
    mocked_pipeline._dump_misc_args()

    # make assertion
    torch_save.assert_called_once_with(
        {
            "data_args": mocked_arguments()[0],
            "model_args": mocked_arguments()[1],
        },
        os.path.join(mocked_arguments()[2].output_dir, "misc_args.bin"),
    )


def test__log_starting_arguments(mocked_arguments, mocker):
    # create mocked pipeline class
    mocked_pipeline = Mocked_Pipeline(*mocked_arguments())

    # mock logger in pipeline
    logger = mocker.patch.object(mocked_pipeline, "logger", create=True)

    # call the pipeline method
    mocked_pipeline._log_starting_arguments()

    # make assertion
    logger.info.assert_called_once()


@pytest.mark.parametrize(
    "seed",
    list(range(5)),
)
def test__set_global_seeds(seed, mocked_arguments, mocker):
    # create mocked pipeline class
    mocked_pipeline = Mocked_Pipeline(*mocked_arguments(seed=seed))

    # mock set_seed in pipeline
    set_seed = mocker.patch("utils.pipeline_utils.set_seed")

    # call the pipeline method
    mocked_pipeline._set_global_seeds()

    # make assertion
    set_seed.assert_called_once_with(seed)


@pytest.mark.parametrize(
    "overwrite_output_dir",
    [True, False],
)
@pytest.mark.parametrize(
    "do_train",
    [True, False],
)
@pytest.mark.parametrize(
    "checkpoint_existing",
    [True, False],
)
def test__find_existing_checkpoint(
    overwrite_output_dir, do_train, checkpoint_existing, mocked_arguments, mocker
):
    # create mocked pipeline class
    mocked_pipeline = Mocked_Pipeline(
        *mocked_arguments(overwrite_output_dir=overwrite_output_dir, do_train=do_train)
    )

    # mock logger in pipeline
    logger = mocker.patch.object(mocked_pipeline, "logger", create=True)

    # mock last_checkpoint function
    if checkpoint_existing:
        get_last_checkpoint = mocker.patch(
            "utils.pipeline_utils.get_last_checkpoint",
            return_value="/path/to/some/checkpoint",
        )
    else:
        get_last_checkpoint = mocker.patch(
            "utils.pipeline_utils.get_last_checkpoint", return_value=None
        )

    # make assertion beforehand
    assert not hasattr(mocked_pipeline, "last_checkpoint")

    # execute relevant pipeline method
    mocked_pipeline._find_existing_checkpoint()

    # make conditional assertions
    if do_train and not overwrite_output_dir:
        get_last_checkpoint.assert_called_once()
        if checkpoint_existing:
            assert mocked_pipeline.last_checkpoint == "/path/to/some/checkpoint"
            logger.warning.assert_called_once()
        else:
            assert mocked_pipeline.last_checkpoint is None
            logger.warning.assert_not_called()
    else:
        assert mocked_pipeline.last_checkpoint is None
        get_last_checkpoint.assert_not_called()
        logger.warning.assert_not_called()


@pytest.mark.parametrize(
    "report_to",
    [["wandb"], []],
)
def test__init_wandb_run(report_to, mocked_arguments, mocker):
    # create mocked pipeline class
    mocked_pipeline = Mocked_Pipeline(*mocked_arguments(report_to=report_to))
    mocked_pipeline.last_checkpoint = "/path/to/some/checkpoint"

    # mock wandb init method
    wandb_init = mocker.patch(
        "utils.pipeline_utils.wandb.init",
    )

    # execute relevant pipeline method
    mocked_pipeline._init_wandb_run()

    # make conditional assertions
    if "wandb" in report_to:
        wandb_init.assert_called_once()
    else:
        wandb_init.assert_not_called()


def test__save_success_file(mocked_arguments_with_tmp_path, mocker):
    # create mocked pipeline class
    mocked_pipeline_with_tmp_path = Mocked_Pipeline(*mocked_arguments_with_tmp_path())

    # execute relevant pipeline method
    mocked_pipeline_with_tmp_path._save_success_file()

    # make assertion
    assert os.path.exists(
        os.path.join(mocked_pipeline_with_tmp_path.train_args.output_dir, ".success")
    )


@pytest.mark.parametrize(
    "has_logger",
    [True, False],
)
def test__clean_loggers(has_logger, mocked_arguments, mocker):
    # create mocked pipeline class
    mocked_pipeline = Mocked_Pipeline(*mocked_arguments())

    # simulate actual loggers
    if has_logger:
        mocked_pipeline.logger = logging.getLogger()
        mocked_pipeline.logger.handlers = ["test"]
    datasets.utils.logging.get_logger().handlers = ["test"]
    transformers.utils.logging.get_logger().handlers = ["test"]

    # execute relevant pipeline method
    mocked_pipeline._clean_loggers()

    # make assertions about root logger
    if has_logger:
        assert mocked_pipeline.logger.handlers == []
    else:
        assert not hasattr(mocked_pipeline, "logger")

    # make assertions about third-party loggers
    assert datasets.utils.logging.get_logger().handlers == []
    assert transformers.utils.logging.get_logger().handlers == []


@pytest.mark.parametrize(
    "report_to",
    [["wandb"], []],
)
@pytest.mark.parametrize(
    "has_wandb_run",
    [True, False],
)
def test__close_wandb(report_to, has_wandb_run, mocked_arguments, mocker):
    # create mocked pipeline class
    mocked_pipeline = Mocked_Pipeline(*mocked_arguments(report_to=report_to))

    # mock wandb module
    wandb = mocker.patch("utils.pipeline_utils.wandb")
    if not has_wandb_run:
        type(wandb).run = mocker.PropertyMock(return_value=None)

    # execute relevant pipeline method
    mocked_pipeline._close_wandb()

    if "wandb" in report_to and has_wandb_run:
        wandb.run.finish.assert_called_once()
    else:
        wandb.assert_not_called()


def test__destroy(mocked_arguments):
    # create mocked pipeline class
    mocked_pipeline = Mocked_Pipeline(*mocked_arguments())
    mocked_pipeline.trainer = []

    # execute relevant pipeline method
    mocked_pipeline._destroy()

    # make assertion
    assert mocked_pipeline.trainer is None


def test_run_start(mocked_arguments, mocker):
    # create mocked pipeline class
    mocked_pipeline = Mocked_Pipeline(*mocked_arguments())

    # create connected mock object
    mock = mocker.MagicMock()
    mock.attach_mock(
        mocker.patch("utils.pipeline_utils.Privacy_GLUE_Pipeline._init_run_dir"),
        "f_1",
    )
    mock.attach_mock(
        mocker.patch("utils.pipeline_utils.Privacy_GLUE_Pipeline._init_root_logger"),
        "f_2",
    )
    mock.attach_mock(
        mocker.patch(
            "utils.pipeline_utils.Privacy_GLUE_Pipeline._init_third_party_loggers"
        ),
        "f_3",
    )
    mock.attach_mock(
        mocker.patch(
            "utils.pipeline_utils.Privacy_GLUE_Pipeline._check_for_success_file"
        ),
        "f_4",
    )
    mock.attach_mock(
        mocker.patch("utils.pipeline_utils.Privacy_GLUE_Pipeline._dump_misc_args"),
        "f_5",
    )
    mock.attach_mock(
        mocker.patch(
            "utils.pipeline_utils.Privacy_GLUE_Pipeline._log_starting_arguments"
        ),
        "f_6",
    )
    mock.attach_mock(
        mocker.patch("utils.pipeline_utils.Privacy_GLUE_Pipeline._set_global_seeds"),
        "f_7",
    )
    mock.attach_mock(
        mocker.patch(
            "utils.pipeline_utils.Privacy_GLUE_Pipeline._find_existing_checkpoint"
        ),
        "f_8",
    )
    mock.attach_mock(
        mocker.patch("utils.pipeline_utils.Privacy_GLUE_Pipeline._init_wandb_run"),
        "f_9",
    )

    # execute relevant pipeline method
    mocked_pipeline.run_start()

    # make assertions
    assert mock.mock_calls == [
        mocker.call.f_1(),
        mocker.call.f_2(),
        mocker.call.f_3(),
        mocker.call.f_4(),
        mocker.call.f_5(),
        mocker.call.f_6(),
        mocker.call.f_7(),
        mocker.call.f_8(),
        mocker.call.f_9(),
    ]


def test_run_task(mocked_arguments, mocker):
    # create mocked pipeline class
    mocked_pipeline = Mocked_Pipeline(*mocked_arguments())

    # create connected mock object
    mock = mocker.MagicMock()
    mock.attach_mock(
        mocker.patch.object(mocked_pipeline, "_retrieve_data"),
        "f_1",
    )
    mock.attach_mock(
        mocker.patch.object(mocked_pipeline, "_load_pretrained_model_and_tokenizer"),
        "f_2",
    )
    mock.attach_mock(
        mocker.patch.object(mocked_pipeline, "_apply_preprocessing"),
        "f_3",
    )
    mock.attach_mock(
        mocker.patch.object(mocked_pipeline, "_set_metrics"),
        "f_4",
    )
    mock.attach_mock(
        mocker.patch.object(mocked_pipeline, "_run_train_loop"),
        "f_5",
    )

    # execute relevant pipeline method
    mocked_pipeline.run_task()

    # make assertions
    assert mock.mock_calls == [
        mocker.call.f_1(),
        mocker.call.f_2(),
        mocker.call.f_3(),
        mocker.call.f_4(),
        mocker.call.f_5(),
    ]


def test_run_end(mocked_arguments, mocker):
    # create mocked pipeline class
    mocked_pipeline = Mocked_Pipeline(*mocked_arguments())

    # mock function
    save_success_file = mocker.patch(
        "utils.pipeline_utils.Privacy_GLUE_Pipeline._save_success_file"
    )

    # execute relevant pipeline method
    mocked_pipeline.run_end()

    # make assertions
    save_success_file.assert_called_once()


def test_run_finally(mocked_arguments, mocker):
    # create mocked pipeline class
    mocked_pipeline = Mocked_Pipeline(*mocked_arguments())

    # create connected mock object
    mock = mocker.MagicMock()
    mock.attach_mock(
        mocker.patch("utils.pipeline_utils.Privacy_GLUE_Pipeline._clean_loggers"),
        "f_1",
    )
    mock.attach_mock(
        mocker.patch("utils.pipeline_utils.Privacy_GLUE_Pipeline._close_wandb"),
        "f_2",
    )
    mock.attach_mock(
        mocker.patch("utils.pipeline_utils.Privacy_GLUE_Pipeline._destroy"),
        "f_3",
    )

    # execute relevant pipeline method
    mocked_pipeline.run_finally()

    # make assertions
    assert mock.mock_calls == [
        mocker.call.f_1(),
        mocker.call.f_2(),
        mocker.call.f_3(),
    ]


@pytest.mark.parametrize(
    "success_file_present",
    [True, False],
)
@pytest.mark.parametrize(
    "error_encountered_in_task",
    [True, False],
)
def test_run_pipeline(
    success_file_present, error_encountered_in_task, mocked_arguments, mocker
):
    # create mocked pipeline class
    mocked_pipeline = Mocked_Pipeline(*mocked_arguments())

    # create mocked methods
    run_start = mocker.patch(
        "utils.pipeline_utils.Privacy_GLUE_Pipeline.run_start",
        side_effect=SuccessFileFoundException if success_file_present else None,
    )
    run_task = mocker.patch(
        "utils.pipeline_utils.Privacy_GLUE_Pipeline.run_task",
        side_effect=ValueError if error_encountered_in_task else None,
    )
    run_end = mocker.patch("utils.pipeline_utils.Privacy_GLUE_Pipeline.run_end")
    run_finally = mocker.patch("utils.pipeline_utils.Privacy_GLUE_Pipeline.run_finally")

    # make assertions
    if not success_file_present:
        if not error_encountered_in_task:
            mocked_pipeline.run_pipeline()
            run_start.assert_called_once()
            run_task.assert_called_once()
            run_end.assert_called_once()
            run_finally.assert_called_once()
        else:
            with pytest.raises(ValueError):
                mocked_pipeline.run_pipeline()
            run_start.assert_called_once()
            run_task.assert_called_once()
            run_end.assert_not_called()
            run_finally.assert_called_once()
    else:
        mocked_pipeline.run_pipeline()
        run_start.assert_called_once()
        run_task.assert_not_called()
        run_end.assert_not_called()
        run_finally.assert_called_once()