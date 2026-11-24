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
from unittest.mock import patch

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
@pytest.mark.parametrize("invalid_cloud", [0, 1])
@pytest.mark.parametrize("invalid_value", [0.0, float("nan"), float("inf"), 1e308])
def test_invalid_point_clouds(return_all, invalid_cloud, invalid_value):
    """Reject an invalid batch member instead of reporting a perfect match for the batch."""
    clouds = [torch.randn(2, 5, 3, dtype=torch.float64) for _ in range(2)]
    clouds[invalid_cloud][1] = invalid_value
    if invalid_value == 1e308:
        clouds[invalid_cloud][1, 0, 0] = -invalid_value
    with pytest.raises(ValueError, match="finite|unique"):
        procrustes_disparity(*clouds, return_all=return_all)


@pytest.mark.parametrize("shape", [(2, 0, 3), (2, 5, 0)])
def test_empty_point_cloud_dimensions(shape):
    """A cloud needs points and coordinate dimensions to be aligned."""
    cloud = torch.empty(shape)
    with pytest.raises(ValueError, match="non-empty"):
        procrustes_disparity(cloud, cloud)


def test_invalid_point_cloud_does_not_update_metric():
    """Invalid updates must not contribute fabricated perfect scores to the running metric."""
    metric = ProcrustesDisparity()
    valid = torch.randn(2, 5, 3)
    metric.update(valid, valid)
    disparity, total = metric.disparity.clone(), metric.total.clone()
    with pytest.raises(ValueError, match="unique"):
        metric.update(torch.ones_like(valid), valid)
    torch.testing.assert_close(metric.disparity, disparity)
    torch.testing.assert_close(metric.total, total)


def test_rank_deficient_point_clouds():
    """Collinear but nonconstant clouds remain valid, including the return_all contract."""
    cloud = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], dtype=torch.float64)
    transformed = 3 * cloud + 4
    disparity, scale, rotation = procrustes_disparity(cloud, transformed, return_all=True)
    torch.testing.assert_close(disparity, torch.zeros(1, dtype=cloud.dtype), atol=1e-12, rtol=0)
    assert scale.shape == (1, 1)
    assert rotation.shape == (1, 3, 3)
    assert scale.dtype == rotation.dtype == cloud.dtype


@pytest.mark.parametrize("return_all", [False, True])
def test_svd_failure_is_not_a_perfect_match(return_all):
    """Preserve backend failures instead of returning fabricated zero disparity."""
    cloud = torch.randn(2, 5, 3)
    with (
        patch("torch.linalg.svd", side_effect=torch.linalg.LinAlgError("SVD did not converge")),
        pytest.raises(torch.linalg.LinAlgError, match="SVD did not converge"),
    ):
        procrustes_disparity(cloud, cloud, return_all=return_all)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("invalid_cloud", [0, 1])
@pytest.mark.parametrize("return_all", [False, True])
def test_nonzero_constant_cloud_with_rounding(dtype, invalid_cloud, return_all):
    """Mean rounding must not turn identical points into an apparently nonzero-spread cloud."""
    clouds = [torch.arange(21, dtype=dtype).reshape(1, 7, 3) for _ in range(2)]
    clouds[invalid_cloud] = torch.full_like(clouds[invalid_cloud], 0.1)
    with pytest.raises(ValueError, match="unique"):
        procrustes_disparity(*clouds, return_all=return_all)
