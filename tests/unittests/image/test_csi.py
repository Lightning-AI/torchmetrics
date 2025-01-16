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

import pytest
import torch
from sklearn.metrics import jaccard_score

from torchmetrics.functional.regression.csi import critical_success_index
from torchmetrics.regression.csi import CriticalSuccessIndex
from unittests import BATCH_SIZE, NUM_BATCHES, _Input
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

seed_all(42)


_inputs_1 = _Input(preds=torch.rand(NUM_BATCHES, BATCH_SIZE), target=torch.rand(NUM_BATCHES, BATCH_SIZE))
_inputs_2 = _Input(preds=torch.rand(NUM_BATCHES, BATCH_SIZE), target=torch.rand(NUM_BATCHES, BATCH_SIZE))


def _reference_sklearn_jaccard(preds: torch.Tensor, target: torch.Tensor, threshold: float):
    """Calculate reference metric for `CriticalSuccessIndex`."""
    preds, target = preds.numpy(), target.numpy()
    preds = preds >= threshold
    target = target >= threshold
    return jaccard_score(preds.ravel(), target.ravel())


@pytest.mark.parametrize(
    "preds, target",
    [
        (_inputs_1.preds, _inputs_1.target),
        (_inputs_2.preds, _inputs_2.target),
    ],
)
@pytest.mark.parametrize("threshold", [0.5, 0.25, 0.75])
class TestCriticalSuccessIndex(MetricTester):
    """Test class for `CriticalSuccessIndex` metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_csi_class(self, preds, target, threshold, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=CriticalSuccessIndex,
            reference_metric=partial(_reference_sklearn_jaccard, threshold=threshold),
            metric_args={"threshold": threshold},
        )

    def test_csi_functional(self, preds, target, threshold):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=critical_success_index,
            reference_metric=partial(_reference_sklearn_jaccard, threshold=threshold),
            metric_args={"threshold": threshold},
        )

    def test_csi_half_cpu(self, preds, target, threshold):
        """Test dtype support of the metric on CPU."""
        self.run_precision_test_cpu(
            preds=preds, target=target, metric_functional=critical_success_index, metric_args={"threshold": threshold}
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_csi_half_gpu(self, preds, target, threshold):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(
            preds=preds, target=target, metric_functional=critical_success_index, metric_args={"threshold": threshold}
        )


def test_error_on_different_shape():
    """Test that error is raised on different shapes of input."""
    metric = CriticalSuccessIndex(0.5)
    with pytest.raises(RuntimeError, match="Predictions and targets are expected to have the same shape"):
        metric(torch.randn(100), torch.randn(50))
