#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from reading_comprehension import Reading_Comprehension_Pipeline
import pytest


def test__retrieve_data(mocked_arguments, mocker):
    # create mocked pipeline object
    mocked_pipeline = Reading_Comprehension_Pipeline(*mocked_arguments())

    # mock relevant method
    get_data = mocker.patch(
        "reading_comprehension.Reading_Comprehension_Pipeline._get_data",
        return_value="mocked_dictionary",
    )

    # execute relevant pipeline method
    mocked_pipeline._retrieve_data()

    # make assertions on changes
    get_data.assert_called_once()
    assert mocked_pipeline.raw_datasets == "mocked_dictionary"


@pytest.mark.parametrize(
    "use_fast",
    [True, False],
)
def test__load_pretrained_model_and_tokenizer(use_fast, mocked_arguments, mocker):
    # create mocked pipeline object
    mocked_pipeline = Reading_Comprehension_Pipeline(*mocked_arguments())

    # mock relevant modules
    auto_config = mocker.patch(
        "reading_comprehension.AutoConfig.from_pretrained",
        return_value="mocked_config",
    )
    auto_tokenizer = mocker.patch(
        "reading_comprehension.AutoTokenizer.from_pretrained",
        return_value=mocker.MagicMock(
            spec=PreTrainedTokenizerFast if use_fast else PreTrainedTokenizer,
            return_value="mocked_tokenizer",
        ),
    )
    auto_model = mocker.patch(
        "reading_comprehension.AutoModelForQuestionAnswering.from_pretrained",
        return_value="mocked_model",
    )

    # execute relevant pipeline method
    if use_fast:
        mocked_pipeline._load_pretrained_model_and_tokenizer()
    else:
        with pytest.raises(ValueError):
            mocked_pipeline._load_pretrained_model_and_tokenizer()

    # make assertions
    auto_config.assert_called_once()
    auto_tokenizer.assert_called_once()
    auto_model.assert_called_once()
    assert (
        auto_config.call_args.args[0]
        == auto_tokenizer.call_args.args[0]
        == auto_model.call_args.args[0]
        == mocked_arguments()[1].model_name_or_path
    )

    assert mocked_pipeline.config == "mocked_config"
    assert mocked_pipeline.tokenizer.return_value == "mocked_tokenizer"
    assert mocked_pipeline.model == "mocked_model"
