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
from copy import deepcopy
from functools import partial
from typing import Any

import pytest
import torch
from torch import Tensor

from torchmetrics.classification import BinaryAccuracy, BinaryConfusionMatrix, MulticlassAccuracy
from torchmetrics.regression import MeanSquaredError
from torchmetrics.wrappers import MinMaxMetric
from unittests import BATCH_SIZE, NUM_BATCHES, NUM_CLASSES
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

seed_all(42)


class TestingMinMaxMetric(MinMaxMetric):
    """Wrap metric to fit testing framework."""

    def compute(self):
        """Instead of returning dict, return as list."""
        output_dict = super().compute()
        return [output_dict["raw"], output_dict["min"], output_dict["max"]]

    def forward(self, *args: Any, **kwargs: Any):
        """Compute output for batch."""
        self.update(*args, **kwargs)
        return self.compute()


def _compare_fn(preds, target, base_fn):
    """Comparison function for minmax wrapper."""
    v_min, v_max = 1e6, -1e6  # pick some very large numbers for comparing
    for i in range(NUM_BATCHES):
        val = base_fn(preds[: (i + 1) * BATCH_SIZE], target[: (i + 1) * BATCH_SIZE]).cpu().numpy()
        v_min = v_min if v_min < val else val
        v_max = v_max if v_max > val else val
    raw = base_fn(preds, target)
    return [raw.cpu().numpy(), v_min, v_max]


@pytest.mark.parametrize(
    ("preds", "target", "base_metric"),
    [
        (
            torch.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES).softmax(dim=-1),
            torch.randint(NUM_CLASSES, (NUM_BATCHES, BATCH_SIZE)),
            MulticlassAccuracy(num_classes=NUM_CLASSES),
        ),
        (torch.randn(NUM_BATCHES, BATCH_SIZE), torch.randn(NUM_BATCHES, BATCH_SIZE), MeanSquaredError()),
    ],
)
class TestMinMaxWrapper(MetricTester):
    """Test the MinMaxMetric wrapper works as expected."""

    atol = 1e-6

    def test_minmax_wrapper(self, preds, target, base_metric):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=False,
            preds=preds,
            target=target,
            metric_class=TestingMinMaxMetric,
            reference_metric=partial(_compare_fn, base_fn=deepcopy(base_metric)),
            metric_args={"base_metric": base_metric},
            check_batch=False,
            check_scriptable=False,
        )


@pytest.mark.parametrize(
    ("preds", "labels", "raws", "maxs", "mins"),
    [
        (
            ([[0.9, 0.1], [0.2, 0.8]], [[0.1, 0.9], [0.2, 0.8]], [[0.1, 0.9], [0.8, 0.2]]),
            [[0, 1], [0, 1]],
            (0.5, 1.0, 0.5),
            (0.5, 1.0, 1.0),
            (0.5, 0.5, 0.5),
        )
    ],
)
def test_basic_example(preds, labels, raws, maxs, mins) -> None:
    """Tests that both min and max versions of MinMaxMetric operate correctly after calling compute."""
    acc = BinaryAccuracy()
    min_max_acc = MinMaxMetric(acc)
    labels = Tensor(labels).long()

    for i in range(3):
        preds_ = Tensor(preds[i])
        min_max_acc(preds_, labels)
        acc = min_max_acc.compute()
        assert acc["raw"] == raws[i]
        assert acc["max"] == maxs[i]
        assert acc["min"] == mins[i]


def test_no_base_metric() -> None:
    """Tests that ValueError is raised when no base_metric is passed."""
    with pytest.raises(ValueError, match=r"Expected base metric to be an instance .*"):
        MinMaxMetric([])


def test_reset_clears_min_max() -> None:
    """Tests that reset() properly resets min_val and max_val to initialization bounds (issue #3328)."""
    min_max_acc: MinMaxMetric = MinMaxMetric(BinaryAccuracy())
    preds = Tensor([[0.9, 0.1], [0.2, 0.8]])
    labels = Tensor([[0, 1], [0, 1]]).long()
    min_max_acc(preds, labels)
    result = min_max_acc.compute()
    assert result["min"].item() != float("inf")
    assert result["max"].item() != float("-inf")

    min_max_acc.reset()
    assert min_max_acc.min_val.item() == float("inf"), "min_val should be reset to inf after reset()"
    assert min_max_acc.max_val.item() == float("-inf"), "max_val should be reset to -inf after reset()"


def test_reset_no_pollution_across_epochs() -> None:
    """Make sure min/max values from a previous epoch do not leak into the next epoch after reset (issue #3328)."""
    min_max_acc: MinMaxMetric = MinMaxMetric(BinaryAccuracy())
    labels = Tensor([[0, 1], [0, 1]]).long()

    # Epoch 1 (e.g. Lightning sanity check): perfect predictions -> accuracy = 1.0
    min_max_acc(Tensor([[0.1, 0.9], [0.1, 0.9]]), labels)
    epoch1 = min_max_acc.compute()  # type: ignore[call-arg]
    assert epoch1["raw"].item() == 1.0
    assert epoch1["max"].item() == 1.0

    min_max_acc.reset()

    # Epoch 2: worse predictions -> accuracy = 0.5; max must not be 1.0 leaked from epoch 1
    min_max_acc(Tensor([[0.9, 0.1], [0.1, 0.9]]), labels)
    epoch2 = min_max_acc.compute()  # type: ignore[call-arg]
    assert epoch2["raw"].item() == 0.5
    assert epoch2["max"].item() == 0.5, "max_val should not retain values from before reset()"
    assert epoch2["min"].item() == 0.5, "min_val should not retain values from before reset()"


def test_no_scalar_compute() -> None:
    """Tests that an assertion error is thrown if the wrapped basemetric gives a non-scalar on compute."""
    min_max_nsm = MinMaxMetric(BinaryConfusionMatrix())

    with pytest.raises(RuntimeError, match=r"Returned value from base metric should be a float.*"):
        min_max_nsm.compute()
