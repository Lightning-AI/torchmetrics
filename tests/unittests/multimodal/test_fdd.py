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
from typing import NamedTuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from torch import Tensor

from torchmetrics.functional.multimodal.fdd import upper_face_dynamics_deviation
from torchmetrics.multimodal.fdd import UpperFaceDynamicsDeviation
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

seed_all(42)


class _InputVertices(NamedTuple):
    vertices_pred: Tensor
    vertices_gt: Tensor


def _generate_vertices(batch_size: int = 1) -> _InputVertices:
    """Generate random vertices for testing."""
    return _InputVertices(
        vertices_pred=torch.randn(batch_size, 10, 100, 3),
        vertices_gt=torch.randn(batch_size, 10, 100, 3),
    )


def _reference_fdd(vertices_pred, vertices_gt, upper_face_map):
    """Reference implementation for FDD metric using numpy."""

    pred = vertices_pred[:, upper_face_map, :].numpy()  # (T, M, 3)
    gt = vertices_gt[:, upper_face_map, :].numpy()      # (T, M, 3)

    displacements_gt = gt[1:] - gt[:-1]  # (T-1, V, 3)
    displacements_pred = pred[1:] - pred[:-1]

    l2_gt = np.linalg.norm(displacements_gt ** 2, axis=-1)  # (T-1, M)
    l2_pred = np.linalg.norm(displacements_pred ** 2, axis=-1)

    std_diff = np.std(l2_gt, axis=0) - np.std(l2_pred, axis=0)  # (M,)

    fdd = np.mean(std_diff)
    
    return torch.tensor(fdd)


class TestUpperFaceDynamicsDeviation(MetricTester):
    """Test class for `UpperFaceDynamicsDeviation` metric (FDD)."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_fdd_metric_class(self, ddp):
        """Test class implementation of metric."""
        upper_face_map = [0, 1, 2, 3, 4]
        vertices_pred, vertices_gt = _generate_vertices(batch_size=4)

        self.run_class_metric_test(
            ddp=ddp,
            preds=vertices_pred,
            target=vertices_gt,
            metric_class=UpperFaceDynamicsDeviation,
            reference_metric=partial(_reference_fdd, upper_face_map=upper_face_map),
            metric_args={"upper_face_map": upper_face_map},
        )

    def test_fdd_functional(self):
        """Test functional implementation of metric."""
        upper_face_map = [0, 1, 2, 3, 4]
        vertices_pred, vertices_gt = _generate_vertices(batch_size=4)

        self.run_functional_metric_test(
            preds=vertices_pred,
            target=vertices_gt,
            metric_functional=upper_face_dynamics_deviation,
            reference_metric=partial(_reference_fdd, upper_face_map=upper_face_map),
            metric_args={"upper_face_map": upper_face_map},
        )

    def test_fdd_differentiability(self):
        """Test differentiability of FDD metric."""
        upper_face_map = [0, 1, 2, 3, 4]
        vertices_pred, vertices_gt = _generate_vertices(batch_size=4)

        self.run_differentiability_test(
            preds=vertices_pred,
            target=vertices_gt,
            metric_module=UpperFaceDynamicsDeviation,
            metric_functional=upper_face_dynamics_deviation,
            metric_args={"upper_face_map": upper_face_map},
        )

    def test_error_on_wrong_dimensions(self):
        """Test that an error is raised for wrong input dimensions."""
        metric = UpperFaceDynamicsDeviation(upper_face_map=[0, 1, 2, 3, 4])
        with pytest.raises(ValueError, match="Expected both vertices_pred and vertices_gt to have 3 dimensions.*"):
            metric(torch.randn(10, 100), torch.randn(10, 100, 3))

    def test_error_on_mismatched_dimensions(self):
        """Test that an error is raised for mismatched vertex dimensions."""
        metric = UpperFaceDynamicsDeviation(upper_face_map=[0, 1, 2, 3, 4])
        with pytest.raises(ValueError, match="Expected vertices_pred and vertices_gt to have same vertex.*"):
            metric(torch.randn(10, 80, 3), torch.randn(10, 100, 3))

    def test_error_on_empty_upper_face_map(self):
        """Test that an error is raised if upper_face_map is empty."""
        with pytest.raises(ValueError, match="upper_face_map cannot be empty."):
            UpperFaceDynamicsDeviation(upper_face_map=[])

    def test_error_on_invalid_upper_face_indices(self):
        """Test that an error is raised if upper_face_map has invalid indices."""
        metric = UpperFaceDynamicsDeviation(upper_face_map=[98, 99, 100])
        with pytest.raises(ValueError, match="upper_face_map contains invalid vertex indices.*"):
            metric(torch.randn(10, 50, 3), torch.randn(10, 50, 3))

    def test_different_sequence_lengths(self):
        """Test that the metric handles sequences of different lengths correctly."""
        metric = UpperFaceDynamicsDeviation(upper_face_map=[0, 1, 2])
        pred = torch.randn(10, 50, 3)
        target = torch.randn(8, 50, 3)
        with pytest.raises(ValueError, match="Expected vertices_pred and vertices_gt to have same vertex.*"):
            metric(pred, target)

    def test_plot_method(self):
        """Test the plot method of FDD."""
        metric = UpperFaceDynamicsDeviation(upper_face_map=[0, 1, 2, 3, 4])
        vertices_pred, vertices_gt = _generate_vertices()
        metric.update(vertices_pred[0], vertices_gt[0])
        fig, ax = metric.plot()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)
