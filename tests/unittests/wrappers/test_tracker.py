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

import pytest
import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassPrecision,
    MulticlassRecall,
)
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
from torchmetrics.wrappers import MetricTracker, MultioutputWrapper

from unittests.helpers import seed_all

seed_all(42)


def test_raises_error_on_wrong_input():
    """Make sure that input type errors are raised on the wrong input."""
    with pytest.raises(TypeError, match="Metric arg need to be an instance of a .*"):
        MetricTracker([1, 2, 3])

    with pytest.raises(ValueError, match="Argument `maximize` should either be a single bool or list of bool"):
        MetricTracker(MeanAbsoluteError(), maximize=2)

    with pytest.raises(
        ValueError, match="The len of argument `maximize` should match the length of the metric collection"
    ):
        MetricTracker(MetricCollection([MeanAbsoluteError(), MeanSquaredError()]), maximize=[False, False, False])

    with pytest.raises(
        ValueError, match="Argument `maximize` should be a single bool when `metric` is a single Metric"
    ):
        MetricTracker(MeanAbsoluteError(), maximize=[False])


@pytest.mark.parametrize(
    ("method", "method_input"),
    [
        ("update", (torch.randint(10, (50,)), torch.randint(10, (50,)))),
        ("forward", (torch.randint(10, (50,)), torch.randint(10, (50,)))),
        ("compute", None),
    ],
)
def test_raises_error_if_increment_not_called(method, method_input):
    """Test that error is raised if another method is called before increment."""
    tracker = MetricTracker(MulticlassAccuracy(num_classes=10))
    with pytest.raises(ValueError, match=f"`{method}` cannot be called before .*"):  # noqa: PT012
        if method_input is not None:
            getattr(tracker, method)(*method_input)
        else:
            getattr(tracker, method)()


@pytest.mark.parametrize(
    ("base_metric", "metric_input", "maximize"),
    [
        (MulticlassAccuracy(num_classes=10), (torch.randint(10, (50,)), torch.randint(10, (50,))), True),
        (MulticlassPrecision(num_classes=10), (torch.randint(10, (50,)), torch.randint(10, (50,))), True),
        (MulticlassRecall(num_classes=10), (torch.randint(10, (50,)), torch.randint(10, (50,))), True),
        (MeanSquaredError(), (torch.randn(50), torch.randn(50)), False),
        (MeanAbsoluteError(), (torch.randn(50), torch.randn(50)), False),
        (
            MetricCollection([
                MulticlassAccuracy(num_classes=10),
                MulticlassPrecision(num_classes=10),
                MulticlassRecall(num_classes=10),
            ]),
            (torch.randint(10, (50,)), torch.randint(10, (50,))),
            True,
        ),
        (
            MetricCollection([
                MulticlassAccuracy(num_classes=10),
                MulticlassPrecision(num_classes=10),
                MulticlassRecall(num_classes=10),
            ]),
            (torch.randint(10, (50,)), torch.randint(10, (50,))),
            [True, True, True],
        ),
        (MetricCollection([MeanSquaredError(), MeanAbsoluteError()]), (torch.randn(50), torch.randn(50)), False),
        (
            MetricCollection([MeanSquaredError(), MeanAbsoluteError()]),
            (torch.randn(50), torch.randn(50)),
            [False, False],
        ),
    ],
)
def test_tracker(base_metric, metric_input, maximize):
    """Test that arguments gets passed correctly to child modules."""
    tracker = MetricTracker(base_metric, maximize=maximize)
    for i in range(5):
        tracker.increment()
        # check both update and forward works
        for _ in range(5):
            tracker.update(*metric_input)
        for _ in range(5):
            tracker(*metric_input)

        # Make sure we have computed something
        val = tracker.compute()
        if isinstance(val, dict):
            for v in val.values():
                assert v != 0.0
        else:
            assert val != 0.0
        assert tracker.n_steps == i + 1

    # Assert that compute all returns all values
    assert tracker.n_steps == 5
    all_computed_val = tracker.compute_all()
    if isinstance(all_computed_val, dict):
        for v in all_computed_val.values():
            assert v.numel() == 5
    else:
        assert all_computed_val.numel() == 5

    # Assert that best_metric returns both index and value
    val, idx = tracker.best_metric(return_step=True)
    if isinstance(val, dict):
        for v, i in zip(val.values(), idx.values()):
            assert v != 0.0
            assert i in list(range(5))
    else:
        assert val != 0.0
        assert idx in list(range(5))

    val2 = tracker.best_metric(return_step=False)
    assert val == val2


@pytest.mark.parametrize(
    "base_metric",
    [
        MulticlassConfusionMatrix(3),
        MetricCollection([MulticlassConfusionMatrix(3), MulticlassAccuracy(3)]),
    ],
)
def test_best_metric_for_not_well_defined_metric_collection(base_metric):
    """Check for user warnings related to best metric.

    Test that if user tries to compute the best metric for a metric that does not have a well defined best, we throw an
    warning and return None.

    """
    tracker = MetricTracker(base_metric)
    for _ in range(3):
        tracker.increment()
        for _ in range(5):
            tracker.update(torch.randint(3, (10,)), torch.randint(3, (10,)))

    with pytest.warns(UserWarning, match="Encountered the following error when trying to get the best metric.*"):
        best = tracker.best_metric()
        if isinstance(best, dict):
            assert best["MulticlassAccuracy"] is not None
            assert best["MulticlassConfusionMatrix"] is None
        else:
            assert best is None

    with pytest.warns(UserWarning, match="Encountered the following error when trying to get the best metric.*"):
        best, idx = tracker.best_metric(return_step=True)

        if isinstance(best, dict):
            assert best["MulticlassAccuracy"] is not None
            assert best["MulticlassConfusionMatrix"] is None
            assert idx["MulticlassAccuracy"] is not None
            assert idx["MulticlassConfusionMatrix"] is None
        else:
            assert best is None
            assert idx is None


@pytest.mark.parametrize(
    ("input_to_tracker", "assert_type"),
    [
        (MultioutputWrapper(MeanSquaredError(), num_outputs=2), torch.Tensor),
        (  # nested version
            MetricCollection({
                "mse": MultioutputWrapper(MeanSquaredError(), num_outputs=2),
                "mae": MultioutputWrapper(MeanAbsoluteError(), num_outputs=2),
            }),
            dict,
        ),
    ],
)
def test_metric_tracker_and_collection_multioutput(input_to_tracker, assert_type):
    """Check that MetricTracker support wrapper inputs and nested structures."""
    tracker = MetricTracker(input_to_tracker)
    for _ in range(5):
        tracker.increment()
        for _ in range(5):
            preds, target = torch.randn(100, 2), torch.randn(100, 2)
            tracker.update(preds, target)
    all_res = tracker.compute_all()
    assert isinstance(all_res, assert_type)
    best_metric, which_epoch = tracker.best_metric(return_step=True)
    if isinstance(best_metric, dict):
        for v in best_metric.values():
            assert v is None
        for v in which_epoch.values():
            assert v is None
    else:
        assert best_metric is None
        assert which_epoch is None
