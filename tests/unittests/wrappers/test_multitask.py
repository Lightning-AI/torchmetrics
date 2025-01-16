# Copyright The Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# this is just a bypass for this module name collision with built-in one
import re

import pytest
import torch

from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_2_5
from torchmetrics.wrappers import MultitaskWrapper
from unittests import BATCH_SIZE, NUM_BATCHES
from unittests._helpers import seed_all

seed_all(42)

_regression_preds = torch.rand(NUM_BATCHES, BATCH_SIZE)
_regression_target = torch.rand(NUM_BATCHES, BATCH_SIZE)
_regression_preds_2 = torch.rand(NUM_BATCHES, BATCH_SIZE)
_regression_target_2 = torch.rand(NUM_BATCHES, BATCH_SIZE)
_classification_preds = torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE))
_classification_target = torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE))
_multitask_preds = {"Classification": _classification_preds, "Regression": _regression_preds}
_multitask_targets = {"Classification": _classification_target, "Regression": _regression_target}


def _dict_results_same_as_individual_results(classification_results, regression_results, multitask_results):
    return (
        multitask_results["Classification"] == classification_results
        and multitask_results["Regression"] == regression_results
    )


def _multitask_same_as_individual_tasks(classification_metric, regression_metric, multitask_metrics):
    """Update classification and regression metrics individually and together using a multitask wrapper.

    Return True if the results are the same.

    """
    classification_metric.update(_classification_preds, _classification_target)
    regression_metric.update(_regression_preds, _regression_target)
    multitask_metrics.update(_multitask_preds, _multitask_targets)

    classification_results = classification_metric.compute()
    regression_results = regression_metric.compute()
    multitask_results = multitask_metrics.compute()

    return _dict_results_same_as_individual_results(classification_results, regression_results, multitask_results)


def test_errors_on_wrong_input():
    """Check that type errors are raised when inputs are of the wrong type."""
    with pytest.raises(TypeError, match="Expected *"):
        MultitaskWrapper(1)

    with pytest.raises(TypeError, match="Expected *"):
        MultitaskWrapper(None)

    with pytest.raises(TypeError, match="Expected *"):
        MultitaskWrapper({"Classification": 1})


def test_error_on_wrong_keys():
    """Check that ValueError is raised when the sets of keys of the task metrics, preds, and targets do not match."""
    multitask_metrics = MultitaskWrapper({
        "Classification": BinaryAccuracy(),
        "Regression": MeanSquaredError(),
    })

    # Classification preds, but not regression preds
    wrong_key_preds = {"Classification": _classification_preds}

    # Classification targets, but not regression targets
    wrong_key_targets = {"Classification": _classification_target}

    # Classification metric, but not regression metric
    wrong_key_multitask_metrics = MultitaskWrapper({
        "Classification": BinaryAccuracy(),
    })

    order_dict = "" if _TORCH_GREATER_EQUAL_2_5 else "o"

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Expected arguments `task_preds` and `task_targets` to have the same keys as the wrapped `task_metrics`."
            " Found task_preds.keys() = dict_keys(['Classification']),"
            " task_targets.keys() = dict_keys(['Classification', 'Regression'])"
            f" and self.task_metrics.keys() = {order_dict}dict_keys(['Classification', 'Regression'])"
        ),
    ):
        multitask_metrics.update(wrong_key_preds, _multitask_targets)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Expected arguments `task_preds` and `task_targets` to have the same keys as the wrapped `task_metrics`."
            " Found task_preds.keys() = dict_keys(['Classification', 'Regression']),"
            " task_targets.keys() = dict_keys(['Classification'])"
            f" and self.task_metrics.keys() = {order_dict}dict_keys(['Classification', 'Regression'])"
        ),
    ):
        multitask_metrics.update(_multitask_preds, wrong_key_targets)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Expected arguments `task_preds` and `task_targets` to have the same keys as the wrapped `task_metrics`."
            " Found task_preds.keys() = dict_keys(['Classification', 'Regression']),"
            " task_targets.keys() = dict_keys(['Classification', 'Regression'])"
            f" and self.task_metrics.keys() = {order_dict}dict_keys(['Classification'])"
        ),
    ):
        wrong_key_multitask_metrics.update(_multitask_preds, _multitask_targets)


def test_basic_multitask():
    """Check that wrapping some Metrics in a MultitaskWrapper is the same as computing them individually."""
    classification_metric = BinaryAccuracy()
    regression_metric = MeanSquaredError()
    multitask_metrics = MultitaskWrapper({"Classification": BinaryAccuracy(), "Regression": MeanSquaredError()})

    assert _multitask_same_as_individual_tasks(classification_metric, regression_metric, multitask_metrics)


def test_metric_collection_multitask():
    """Check that wrapping some MetricCollections in a MultitaskWrapper is the same as computing them individually."""
    classification_metric = MetricCollection([BinaryAccuracy(), BinaryF1Score()])
    regression_metric = MetricCollection([MeanSquaredError(), MeanAbsoluteError()])
    multitask_metrics = MultitaskWrapper({
        "Classification": MetricCollection([BinaryAccuracy(), BinaryF1Score()]),
        "Regression": MetricCollection([MeanSquaredError(), MeanAbsoluteError()]),
    })

    assert _multitask_same_as_individual_tasks(classification_metric, regression_metric, multitask_metrics)


def test_forward():
    """Check that the forward method works as expected."""
    classification_metric = BinaryAccuracy()
    regression_metric = MeanSquaredError()
    multitask_metrics = MultitaskWrapper({"Classification": BinaryAccuracy(), "Regression": MeanSquaredError()})

    classification_results = classification_metric(_classification_preds, _classification_target)
    regression_results = regression_metric(_regression_preds, _regression_target)
    multitask_results = multitask_metrics(_multitask_preds, _multitask_targets)

    assert _dict_results_same_as_individual_results(classification_results, regression_results, multitask_results)


def test_nested_multitask_wrapper():
    """Check that nested multitask wrappers work as expected."""
    classification_metric = BinaryAccuracy()
    regression_position_metric = MeanSquaredError()
    regression_size_metric = MeanAbsoluteError()
    multitask_metrics = MultitaskWrapper({
        "Classification": BinaryAccuracy(),
        "Regression": MultitaskWrapper({
            "Position": MeanSquaredError(),
            "Size": MeanAbsoluteError(),
        }),
    })

    multitask_preds = {
        "Classification": _classification_preds,
        "Regression": {
            "Position": _regression_preds,
            "Size": _regression_preds_2,
        },
    }

    multitask_targets = {
        "Classification": _classification_target,
        "Regression": {
            "Position": _regression_target,
            "Size": _regression_target_2,
        },
    }

    classification_metric.update(_classification_preds, _classification_target)
    regression_position_metric.update(_regression_preds, _regression_target)
    regression_size_metric.update(_regression_preds_2, _regression_target_2)
    multitask_metrics.update(multitask_targets, multitask_preds)

    classification_results = classification_metric.compute()
    regression_position_results = regression_position_metric.compute()
    regression_size_results = regression_size_metric.compute()
    regression_results = {"Position": regression_position_results, "Size": regression_size_results}
    multitask_results = multitask_metrics.compute()

    assert _dict_results_same_as_individual_results(classification_results, regression_results, multitask_results)


@pytest.mark.parametrize("method", ["keys", "items", "values"])
@pytest.mark.parametrize("flatten", [True, False])
def test_key_value_items_method(method, flatten):
    """Test the keys, items, and values methods of the MultitaskWrapper."""
    multitask = MultitaskWrapper({
        "classification": MetricCollection([BinaryAccuracy(), BinaryF1Score()]),
        "regression": MetricCollection([MeanSquaredError(), MeanAbsoluteError()]),
    })
    if method == "keys":
        output = list(multitask.keys(flatten=flatten))
    elif method == "items":
        output = list(multitask.items(flatten=flatten))
    elif method == "values":
        output = list(multitask.values(flatten=flatten))

    if flatten:
        assert len(output) == 4
        if method == "keys":
            assert output == [
                "classification_BinaryAccuracy",
                "classification_BinaryF1Score",
                "regression_MeanSquaredError",
                "regression_MeanAbsoluteError",
            ]
        elif method == "items":
            assert output == [
                ("classification_BinaryAccuracy", BinaryAccuracy()),
                ("classification_BinaryF1Score", BinaryF1Score()),
                ("regression_MeanSquaredError", MeanSquaredError()),
                ("regression_MeanAbsoluteError", MeanAbsoluteError()),
            ]
        elif method == "values":
            assert output == [BinaryAccuracy(), BinaryF1Score(), MeanSquaredError(), MeanAbsoluteError()]
    else:
        assert len(output) == 2
        if method == "keys":
            assert output == ["classification", "regression"]
        elif method == "items":
            assert output[0][0] == "classification"
            assert output[1][0] == "regression"
            assert isinstance(output[0][1], MetricCollection)
            assert isinstance(output[1][1], MetricCollection)
        elif method == "values":
            assert isinstance(output[0], MetricCollection)
            assert isinstance(output[1], MetricCollection)


def test_clone_with_prefix_and_postfix():
    """Check that the clone method works with prefix and postfix arguments."""
    multitask_metrics = MultitaskWrapper(
        {"Classification": BinaryAccuracy(), "Regression": MeanSquaredError()},
        prefix="prefix_",
        postfix="_postfix",
    )
    assert set(multitask_metrics.keys()) == {"prefix_Classification_postfix", "prefix_Regression_postfix"}

    output = multitask_metrics(
        {"Classification": _classification_preds, "Regression": _regression_preds},
        {"Classification": _classification_target, "Regression": _regression_target},
    )
    assert set(output.keys()) == {"prefix_Classification_postfix", "prefix_Regression_postfix"}

    cloned_metrics = multitask_metrics.clone(prefix="new_prefix_", postfix="_new_postfix")
    assert set(cloned_metrics.keys()) == {"new_prefix_Classification_new_postfix", "new_prefix_Regression_new_postfix"}

    output = cloned_metrics(
        {"Classification": _classification_preds, "Regression": _regression_preds},
        {"Classification": _classification_target, "Regression": _regression_target},
    )
    assert set(output.keys()) == {"new_prefix_Classification_new_postfix", "new_prefix_Regression_new_postfix"}
