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

from torchmetrics.functional.shape.procrustes import procrustes_disparity
from torchmetrics.shape.procrustes import ProcrustesDisparity
from unittests import BATCH_SIZE, EXTRA_DIM, NUM_BATCHES, _Input
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

seed_all(42)

NUM_TARGETS = 5


_inputs = _Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, 50, EXTRA_DIM),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE, 50, EXTRA_DIM),
)


def _reference_procrustes(point_cloud1, point_cloud2, reduction=None):
    point_cloud1 = point_cloud1.numpy()
    point_cloud2 = point_cloud2.numpy()

    if reduction is None:
        return np.array([scipy_procrustes(d1, d2)[2] for d1, d2 in zip(point_cloud1, point_cloud2)])

    disparity = 0
    for d1, d2 in zip(point_cloud1, point_cloud2):
        disparity += scipy_procrustes(d1, d2)[2]
    if reduction == "mean":
        return disparity / len(point_cloud1)
    return disparity


@pytest.mark.parametrize(("point_cloud1", "point_cloud2"), [(_inputs.preds, _inputs.target)])
class TestProcrustesDisparity(MetricTester):
    """Test class for `ProcrustesDisparity` metric."""

    @pytest.mark.parametrize("reduction", ["sum", "mean"])
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_procrustes_disparity(self, reduction, point_cloud1, point_cloud2, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            point_cloud1,
            point_cloud2,
            ProcrustesDisparity,
            partial(_reference_procrustes, reduction=reduction),
            metric_args={"reduction": reduction},
        )

    def test_procrustes_disparity_functional(self, point_cloud1, point_cloud2):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            point_cloud1,
            point_cloud2,
            procrustes_disparity,
            _reference_procrustes,
        )


def test_error_on_different_shape():
    """Test that error is raised on different shapes of input."""
    metric = ProcrustesDisparity()
    with pytest.raises(RuntimeError, match="Predictions and targets are expected to have the same shape"):
        metric(torch.randn(10, 100, 2), torch.randn(10, 50, 2))
    with pytest.raises(RuntimeError, match="Predictions and targets are expected to have the same shape"):
        procrustes_disparity(torch.randn(10, 100, 2), torch.randn(10, 50, 2))


def test_error_on_non_3d_input():
    """Test that error is raised if input is not 3-dimensional."""
    metric = ProcrustesDisparity()
    with pytest.raises(ValueError, match="Expected both datasets to be 3D tensors of shape"):
        metric(torch.randn(100), torch.randn(100))
    with pytest.raises(ValueError, match="Expected both datasets to be 3D tensors of shape"):
        procrustes_disparity(torch.randn(100), torch.randn(100))


@pytest.mark.parametrize("return_all", [False, True])
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_svd_failure_preserves_output_contract(monkeypatch, return_all, batch_size, dtype):
    """SVD fallback must preserve the requested result type, batch dimensions, dtype, and device."""

    def _raise_svd_error(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated SVD failure")

    monkeypatch.setattr(torch.linalg, "svd", _raise_svd_error)
    data = torch.randn(batch_size, 5, 2, dtype=dtype)
    with pytest.warns(UserWarning, match="SVD calculation.*failed"):
        result = procrustes_disparity(data, data, return_all=return_all)
    expected = torch.zeros(batch_size, dtype=dtype)
    if return_all:
        disparity, scale, rotation = result
        torch.testing.assert_close(disparity, expected)
        torch.testing.assert_close(scale, torch.ones(batch_size, 1, dtype=dtype))
        torch.testing.assert_close(rotation, torch.eye(2, dtype=dtype).repeat(batch_size, 1, 1))
    else:
        torch.testing.assert_close(result, expected)
    metric = ProcrustesDisparity().to(dtype)
    with pytest.warns(UserWarning, match="SVD calculation.*failed"):
        metric.update(data, data)
    torch.testing.assert_close(metric.compute(), torch.tensor(0.0, dtype=dtype))
    assert metric.total == batch_size
