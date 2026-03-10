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

import pysdtw
import pytest
import torch
from unittests import BATCH_SIZE, NUM_BATCHES, _Input
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

from torchmetrics.functional.timeseries.softdtw import soft_dtw
from torchmetrics.timeseries.softdtw import SoftDTW

seed_all(42)

_inputs = _Input(
    preds=torch.randn(NUM_BATCHES, BATCH_SIZE, 20, 3, dtype=torch.float64),
    target=torch.randn(NUM_BATCHES, BATCH_SIZE, 30, 3, dtype=torch.float64),
)


def _reference_softdtw(
    preds: torch.Tensor, target: torch.Tensor, gamma: float = 1.0, distance_fn=None, reduction: str = "mean"
) -> torch.Tensor:
    """Reference implementation using tslearn's soft-DTW."""
    preds = preds.to("cpu")
    target = target.to("cpu")
    sdtw = pysdtw.SoftDTW(gamma=gamma, dist_func=distance_fn, use_cuda=False)
    if reduction == "mean":
        return sdtw(preds, target).mean()
    if reduction == "sum":
        return sdtw(preds, target).sum()
    return sdtw(preds, target)


def euclidean_distance(x, y):
    """Squared Euclidean distance."""
    return torch.cdist(x, y, p=2)


def manhattan_distance(x, y):
    """L1 (Manhattan) distance."""
    return torch.cdist(x, y, p=1)


def cosine_distance(x, y):
    """Cosine distance."""
    x_norm = x / x.norm(dim=-1, keepdim=True)
    y_norm = y / y.norm(dim=-1, keepdim=True)
    return 1 - torch.matmul(x_norm, y_norm.transpose(-1, -2))


@pytest.mark.parametrize(("preds", "target"), [(_inputs.preds, _inputs.target)])
class TestSoftDTW(MetricTester):
    """Test class for `SoftDTW` metric."""

    @pytest.mark.parametrize("gamma", [0.1, 0.5, 1.0])
    @pytest.mark.parametrize("distance_fn", [euclidean_distance, manhattan_distance, cosine_distance])
    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_softdtw_class(self, gamma, preds, target, distance_fn, reduction, ddp):
        """Test class implementation of SoftDTW."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            SoftDTW,
            partial(_reference_softdtw, gamma=gamma, distance_fn=distance_fn, reduction=reduction),
            metric_args={"gamma": gamma, "distance_fn": distance_fn, "reduction": reduction},
        )

    @pytest.mark.parametrize("gamma", [0.1, 0.5, 1.0])
    @pytest.mark.parametrize("distance_fn", [euclidean_distance, manhattan_distance, cosine_distance])
    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    def test_softdtw_functional(self, preds, target, gamma, distance_fn, reduction):
        """Test functional implementation of SoftDTW."""
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=soft_dtw,
            reference_metric=partial(_reference_softdtw, gamma=gamma, distance_fn=distance_fn, reduction=reduction),
            metric_args={"gamma": gamma, "distance_fn": distance_fn, "reduction": reduction},
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test is too slow without gpu")
    def test_softdtw_differentiability(self, preds, target):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=SoftDTW,
            metric_functional=soft_dtw,
            metric_args={"gamma": 1.0},
        )


def test_wrong_dimensions():
    """Test that an error is raised if input tensors have wrong dimensions."""
    metric = SoftDTW()
    with pytest.raises(ValueError, match="Inputs preds and target must be 3-dimensional tensors of shape*"):
        metric(torch.randn(10, 100), torch.randn(10, 100, 3))


def test_mismatched_dimensions():
    """Test that an error is raised if input dimensions don't match."""
    metric = SoftDTW()
    with pytest.raises(ValueError, match="Batch size of preds and target must be the same.*"):
        metric(torch.randn(10, 80, 3), torch.randn(12, 100, 3))


def test_mismatched_feature_dimensions():
    """Test that an error is raised if input feature dimensions don't match."""
    metric = SoftDTW()
    with pytest.raises(ValueError, match="Feature dimension of preds and target must be the same.*"):
        metric(torch.randn(10, 80, 3), torch.randn(10, 100, 4))


def test_invalid_gamma():
    """Test that an error is raised if gamma is not a positive float."""
    with pytest.raises(ValueError, match="Argument `gamma` must be a positive float, got -1.0*"):
        SoftDTW(gamma=-1.0)


def test_warning_on_cpu():
    """Test that a warning is raised if SoftDTW is used on CPU."""
    if torch.cuda.is_available():
        pytest.skip("Test only runs on CPU.")
    with pytest.warns(UserWarning, match="SoftDTW is slow on CPU. Consider using a GPU.*"):
        SoftDTW()


def test_invalid_reduction():
    """Test that an error is raised if reduction is not one of [``sum``, ``mean``, ``none``]."""
    with pytest.raises(ValueError, match="Argument `reduction` must be one of .*"):
        SoftDTW(reduction="invalid")
