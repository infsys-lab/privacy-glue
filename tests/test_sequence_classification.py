import pytest
import random
import wandb
import torch
from collections import defaultdict
from torch import nn

from sequence_classification import Sequence_Classification_Pipeline
from parser import DataArguments, ModelArguments

import numpy as np

from transformers import (
    TrainingArguments,
    EvalPrediction,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

import datasets

MOCK_DATA = {
    "multi_label": {
        "text": ["hello world in {}", "hello universe in {}"],
        "label": [[1, 0], [0, 1]],
    },
    "single_label": {
        "text": ["hello world in {}", "hello universe in {}"],
        "label": ["1", "2"],
    },
}

MOCK_RESULTS = {
    "multi_label": EvalPrediction(
        predictions=np.array(
            [
                [-3.3, 3.5, 3.0, -3.5, 3.0, 3.0, -3.0, 3.0, 3.0, -3.0, 3.0, 3.0],
                [-3.3, 3.5, 3.0, -3.5, 3.0, 3.0, -3.0, 3.0, 3.0, -3.0, 3.0, 3.0],
                [-3.3, 3.5, 3.0, -3.5, 3.0, 3.0, -3.0, 3.0, 3.0, -3.0, 3.0, 3.0],
                [-3.3, 3.5, 3.0, -3.5, 3.0, 3.0, -3.0, 3.0, 3.0, -3.0, 3.0, 3.0],
            ]
        ),
        label_ids=np.array(
            [
                [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
                [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
                [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
                [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            ]
        ),
    ),
    "single_label": EvalPrediction(
        predictions=np.array(
            [[1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, 1.0, -1.0, -1.0, -1.0]]
        ),
        label_ids=np.array([0, 1]),
    ),
}


def mock_dataset(problem_type):
    combined = datasets.DatasetDict()
    for split in ["train", "validation", "test"]:
        data = MOCK_DATA[problem_type]
        data["text"] = [d.format(split) for d in data["text"]]
        combined[split] = datasets.Dataset.from_dict(MOCK_DATA[problem_type])
        label_info = datasets.ClassLabel(num_classes=2, names=[1, 2])
        combined[split].features["label"] = label_info
    return combined


class MockTokenizer(PreTrainedTokenizer):
    def __init__(self, model_input_names=["labels", "label_ids"]):
        super().__init__()
        self.lookup = defaultdict(int)

    def _tokenize(self, s):
        return s.split()

    def _convert_token_to_id(self, token):
        return self.lookup[token]


class MockConfig(PretrainedConfig):
    def __init__(self, a=0, b=0, double_output=False, random_torch=True, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b
        self.double_output = double_output
        self.random_torch = random_torch
        self.hidden_size = 1


class MockPreTrainedModel(PreTrainedModel):
    config_class = MockConfig
    base_model_prefix = "regression"

    def __init__(self, config):
        super().__init__(config)
        self.a = nn.Parameter(torch.tensor(config.a).float())
        self.b = nn.Parameter(torch.tensor(config.b).float())
        self.random_torch = config.random_torch

    def forward(self, input_x, labels=None, text=None, **kwargs):
        y = input_x * self.a + self.b
        if self.random_torch:
            torch_rand = torch.randn(1).squeeze()
        np_rand = np.random.rand()
        rand_rand = random.random()

        if self.random_torch:
            y += 0.05 * torch_rand
        y += 0.05 * torch.tensor(np_rand + rand_rand)

        if labels is None:
            return (y,)
        loss = nn.functional.mse_loss(y, labels)
        return (loss, y)


class DummyModel(PreTrainedModel):
    def __init__(self, num_params: int):
        super().__init__(PretrainedConfig())

        self.params = nn.Parameter(torch.randn(num_params))

    def forward(self, input_ids, labels=None):
        if labels is not None:
            return torch.tensor(0.0, device=input_ids.device), input_ids
        else:
            return input_ids


@pytest.fixture(
    scope="module",
    params=["opp_115", "privacy_qa"],
)
def setup_classifier(request):
    wandb.init(mode="disabled")

    pipeline = Sequence_Classification_Pipeline(
        DataArguments(**{"task": request.param}),
        ModelArguments(**{"model_name_or_path": "bert-base-uncased"}),
        TrainingArguments(
            "runs", num_train_epochs=1, do_train=True, do_eval=True, do_predict=True
        ),
    )

    def mock_get_data():
        return mock_dataset(pipeline.problem_type)

    pipeline._get_data = mock_get_data
    yield pipeline
    pipeline._destroy()


def test_data_collection(setup_classifier, mocker):
    # mocker.patch(
    #     "self._get_data",
    #     return_value=mock_dataset(setup_classifier.problem_type),
    # )
    setup_classifier._retrieve_data()
    if setup_classifier.train_args.do_train:
        assert len(setup_classifier.train_dataset) > 0
    if setup_classifier.train_args.do_eval:
        assert len(setup_classifier.eval_dataset) > 0
    if setup_classifier.train_args.do_predict:
        assert len(setup_classifier.predict_dataset) > 0

    if setup_classifier.train_args.do_train and setup_classifier.train_args.do_eval:
        assert setup_classifier.train_dataset != setup_classifier.eval_dataset

    if setup_classifier.train_args.do_train and setup_classifier.train_args.do_predict:
        assert setup_classifier.train_dataset != setup_classifier.predict_dataset


# def test_load_model(setup_classifier, mocker):
#     config = MockConfig()
#     mocker.patch(
#         "transformers.AutoConfig.from_pretrained",
#         return_value=config,
#     )
#     mocker.patch(
#         "transformers.AutoModelForSequenceClassification.from_pretrained",
#         return_value=MockPreTrainedModel(config),
#     )
#     mocker.patch(
#         "transformers.AutoTokenizer.from_pretrained",
#         return_value=MockTokenizer(),
#     )

#     setup_classifier.load_pretrained_model_and_tokenizer()
#     assert setup_classifier.config!=None
#     assert setup_classifier.tokenizer!=None
#     assert setup_classifier.model!=None


def test_preprocessing(setup_classifier, mocker):
    # mocker.patch(
    #     "_get_data",
    #     return_value=mock_dataset(setup_classifier.problem_type),
    # )
    setup_classifier.tokenizer = MockTokenizer()

    setup_classifier._retrieve_data()
    setup_classifier._apply_preprocessing()
    assert True


def test_set_metrics(setup_classifier):
    setup_classifier._set_metrics()

    mock_results = MOCK_RESULTS[setup_classifier.problem_type]
    f1 = setup_classifier.compute_metrics(mock_results)
    if setup_classifier.problem_type == "multi_label":
        assert f1["f1"] == 0.75
    else:
        assert f1["f1"] == 1.0


# def test_train_loop(setup_classifier, mocker):
#     mocker.patch(
#         "utils.data_utils.get_data",
#         return_value=mock_dataset(setup_classifier.problem_type),
#     )
#     setup_classifier.retrieve_data()
#     setup_classifier.config = MockConfig()
#     setup_classifier.model = DummyModel(5)
#     setup_classifier.tokenizer = MockTokenizer()
#     setup_classifier.set_metrics()
#     setup_classifier.run_train_loop()
#     #assert setup_classifier.eval_metrics["eval_f1"] != None
#     assert setup_classifier.predict_metrics["predict_f1"] != None
