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
from unittests import _Input
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

from torchmetrics.functional.timeseries.softdtw import soft_dtw
from torchmetrics.timeseries.softdtw import SoftDTW

seed_all(42)

num_batches = 1
batch_size = 1
_inputs = _Input(
    preds=torch.randn(num_batches, batch_size, 15, 3, dtype=torch.float64),
    target=torch.randn(num_batches, batch_size, 14, 3, dtype=torch.float64),
)


def _reference_softdtw(preds: torch.Tensor, target: torch.Tensor, gamma: float = 1.0, distance_fn=None) -> torch.Tensor:
    """Reference implementation using tslearn's soft-DTW."""
    preds = preds.to("cuda" if torch.cuda.is_available() else "cpu")
    target = target.to("cuda" if torch.cuda.is_available() else "cpu")
    sdtw = pysdtw.SoftDTW(gamma=gamma, dist_func=distance_fn, use_cuda=True if torch.cuda.is_available() else False)
    return sdtw(preds, target)


def euclidean_distance(x, y):
    return torch.cdist(x, y, p=2)


def manhattan_distance(x, y):
    return torch.cdist(x, y, p=1)


def cosine_distance(x, y):
    x_norm = x / x.norm(dim=-1, keepdim=True)
    y_norm = y / y.norm(dim=-1, keepdim=True)
    return 1 - torch.matmul(x_norm, y_norm.transpose(-1, -2))


@pytest.mark.parametrize("preds, target", [(_inputs.preds, _inputs.target)])
class TestSoftDTW(MetricTester):
    """Test class for `SoftDTW` metric."""

    @pytest.mark.parametrize("gamma", [0.1, 0.5, 1.0])
    @pytest.mark.parametrize("distance_fn", [euclidean_distance, manhattan_distance, cosine_distance])
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_softdtw_class(self, gamma, preds, target, distance_fn, ddp):
        """Test class implementation of SoftDTW."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            SoftDTW,
            partial(_reference_softdtw, gamma=gamma, distance_fn=distance_fn),
            metric_args={"gamma": gamma, "distance_fn": distance_fn},
        )

    @pytest.mark.parametrize("gamma", [0.1, 0.5, 1.0])
    @pytest.mark.parametrize("distance_fn", [euclidean_distance, manhattan_distance, cosine_distance])
    def test_softdtw_functional(self, preds, target, gamma, distance_fn):
        """Test functional implementation of SoftDTW."""
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=soft_dtw,
            reference_metric=partial(_reference_softdtw, gamma=gamma, distance_fn=distance_fn),
            metric_args={"gamma": gamma, "distance_fn": distance_fn},
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_softdtw_differentiability(self, preds, target):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=SoftDTW,
            metric_functional=soft_dtw,
            metric_args={"gamma": 0.1},
        )
