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
from functools import partial
from typing import Any

import pytest
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score as sk_r2score
from torch import Tensor, tensor
from torchmetrics import Metric
from torchmetrics.classification import ConfusionMatrix, MulticlassAccuracy
from torchmetrics.regression import R2Score
from torchmetrics.wrappers.multioutput import MultioutputWrapper

from unittests import BATCH_SIZE, NUM_BATCHES, NUM_CLASSES, _Input
from unittests.helpers import seed_all
from unittests.helpers.testers import MetricTester

seed_all(42)


class _MultioutputMetric(Metric):
    """Test class that allows passing base metric as a class rather than its instantiation to the wrapper."""

    def __init__(
        self,
        base_metric_class,
        num_outputs: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.metric = MultioutputWrapper(
            base_metric_class(**kwargs),
            num_outputs=num_outputs,
        )

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update the each pair of outputs and predictions."""
        return self.metric.update(preds, target)

    def compute(self) -> Tensor:
        """Compute the R2 score between each pair of outputs and predictions."""
        return self.metric.compute()

    @torch.jit.unused
    def forward(self, *args: Any, **kwargs: Any):
        """Run forward on the underlying metric."""
        return self.metric(*args, **kwargs)

    def reset(self) -> None:
        """Reset the underlying metric state."""
        self.metric.reset()


num_targets = 2


_multi_target_regression_inputs = _Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, num_targets),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE, num_targets),
)
_multi_target_classification_inputs = _Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, num_targets),
    target=torch.randint(NUM_CLASSES, (NUM_BATCHES, BATCH_SIZE, num_targets)),
)


def _multi_target_sk_r2score(preds, target, adjusted=0, multioutput="raw_values"):
    """Compute R2 score over multiple outputs."""
    sk_preds = preds.view(-1, num_targets).numpy()
    sk_target = target.view(-1, num_targets).numpy()
    r2_score = sk_r2score(sk_target, sk_preds, multioutput=multioutput)
    if adjusted != 0:
        return 1 - (1 - r2_score) * (sk_preds.shape[0] - 1) / (sk_preds.shape[0] - adjusted - 1)
    return r2_score


def _multi_target_sk_accuracy(preds, target, num_outputs):
    """Compute accuracy over multiple outputs."""
    return [accuracy_score(torch.argmax(preds[:, :, i], dim=1), target[:, i]) for i in range(num_outputs)]


@pytest.mark.parametrize(
    "base_metric_class, compare_metric, preds, target, num_outputs",
    [
        (
            R2Score,
            _multi_target_sk_r2score,
            _multi_target_regression_inputs.preds,
            _multi_target_regression_inputs.target,
            num_targets,
        ),
        (
            partial(MulticlassAccuracy, num_classes=NUM_CLASSES, average="micro"),
            partial(_multi_target_sk_accuracy, num_outputs=2),
            _multi_target_classification_inputs.preds,
            _multi_target_classification_inputs.target,
            num_targets,
        ),
    ],
)
class TestMultioutputWrapper(MetricTester):
    """Test the MultioutputWrapper class with regression and classification inner metrics."""

    @pytest.mark.parametrize("ddp", [True, False])
    def test_multioutput_wrapper(self, base_metric_class, compare_metric, preds, target, num_outputs, ddp):
        """Test correctness of implementation.

        Tests that the multioutput wrapper properly slices and computes outputs along the output dimension for both
        classification and regression metrics, by comparing to the metric if they had been calculated sequentially.

        """
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            _MultioutputMetric,
            compare_metric,
            metric_args={"num_outputs": num_outputs, "base_metric_class": base_metric_class},
        )


def test_reset_called_correctly():
    """Check that underlying metric is being correctly reset when calling forward."""
    base_metric = ConfusionMatrix(task="multiclass", num_classes=2)
    cf = MultioutputWrapper(base_metric, num_outputs=2)

    res = cf(tensor([[0, 0]]), tensor([[0, 0]]))
    assert torch.allclose(res[0], tensor([[1, 0], [0, 0]]))
    assert torch.allclose(res[1], tensor([[1, 0], [0, 0]]))
    cf.reset()
    res = cf(tensor([[1, 1]]), tensor([[0, 0]]))
    assert torch.allclose(res[0], tensor([[0, 1], [0, 0]]))
    assert torch.allclose(res[1], tensor([[0, 1], [0, 0]]))


def test_squeeze_argument():
    """Test that the squeeze_outputs argument works as expected."""
    m = MultioutputWrapper(ConfusionMatrix(task="binary"), num_outputs=3)
    m.update(torch.randint(2, (10, 3)), torch.randint(2, (10, 3)))  # as args
    m.update(preds=torch.randint(2, (10, 3)), target=torch.randint(2, (10, 3)))  # as kwargs
    val = m.compute()
    assert val.shape == (3, 2, 2)
