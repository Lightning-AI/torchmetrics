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

import numpy as np
import pytest
import torch
from scipy.spatial import procrustes as scipy_procrustes
from torchmetrics.functional.other.procrustes import procrustes_disparity
from torchmetrics.other.procrustes import ProcrustesDisparity

from unittests import BATCH_SIZE, NUM_BATCHES, _Input, EXTRA_DIM
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

seed_all(42)

NUM_TARGETS = 5


_inputs = _Input(
    dataset1=torch.rand(NUM_BATCHES, BATCH_SIZE, 50, EXTRA_DIM),
    dataset2=torch.rand(NUM_BATCHES, BATCH_SIZE, 50, EXTRA_DIM),
)

def _reference_procrustes(dataset1, dataset2, reduction):
    dataset1 = dataset1.numpy()
    dataset2 = dataset2.numpy()

    if reduction is None:
        return np.array([scipy_procrustes(d1, d2)[2] for d1, d2 in zip(dataset1, dataset2)])

    disparity = 0
    for d1, d2 in zip(dataset1, dataset2):
        disparity += procrustes_disparity(d1, d2)
    if reduction == "mean":
        return disparity / len(dataset1)
    return disparity



@pytest.mark.parametrize("dataset1, dataset2", [(_inputs.dataset1, _inputs.dataset2)])
class TestProcrustesDisparity(MetricTester):
    """Test class for `CosineSimilarity` metric."""
    @pytest.mark.parametrize("reduction", ["sum", "mean"])
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_cosine_similarity(self, reduction, preds, target, ref_metric, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            CosineSimilarity,
            partial(ref_metric, reduction=reduction),
            metric_args={"reduction": reduction},
        )

    def test_cosine_similarity_functional(self, reduction, preds, target, ref_metric):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            cosine_similarity,
            partial(ref_metric, reduction=reduction),
            metric_args={"reduction": reduction},
        )


def test_error_on_different_shape(metric_class=CosineSimilarity):
    """Test that error is raised on different shapes of input."""
    metric = metric_class()
    with pytest.raises(RuntimeError, match="Predictions and targets are expected to have the same shape"):
        metric(torch.randn(100, 2), torch.randn(50, 2))


def test_error_on_non_2d_input():
    """Test that error is raised if input is not 2-dimensional."""
    metric = CosineSimilarity()
    with pytest.raises(ValueError, match="Expected input to cosine similarity to be 2D tensors of shape.*"):
        metric(torch.randn(100), torch.randn(100))
