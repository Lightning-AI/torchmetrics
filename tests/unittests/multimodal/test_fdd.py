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
from unittests import BATCH_SIZE, NUM_BATCHES

seed_all(42)


class _InputVertices(NamedTuple):
    vertices_pred: Tensor
    vertices_gt: Tensor
    template: Tensor


def _generate_vertices() -> _InputVertices:
    """Generate random vertices for testing."""
    return _InputVertices(
        vertices_pred=torch.randn(NUM_BATCHES, BATCH_SIZE, 10, 100, 3),
        vertices_gt=torch.randn(NUM_BATCHES, BATCH_SIZE, 10, 100, 3),
        template=torch.randn(100, 3),
    )


def _reference_fdd(vertices_pred, vertices_gt, template, upper_face_map, is_reduced=False):
    """Reference implementation for FDD metric using numpy."""
    min_frames = min(vertices_pred.shape[1], vertices_gt.shape[1])
    pred = vertices_pred[:, :min_frames, upper_face_map, :].detach().cpu().numpy()  # (B, T, M, 3)
    gt = vertices_gt[:, :min_frames, upper_face_map, :].detach().cpu().numpy()  # (B, T, M, 3)
    template = template[upper_face_map, :].detach().cpu().numpy()  # (M, 3)

    displacements_gt = gt - template  # (B, T, M, 3)
    displacements_pred = pred - template

    l2_gt = np.sum(displacements_gt**2, axis=-1)  # (B, T, M), squared L2 norm
    l2_pred = np.sum(displacements_pred**2, axis=-1)

    std_diff = np.std(l2_gt, axis=1) - np.std(l2_pred, axis=1)  # (B, M)

    fdd = np.mean(std_diff, axis=-1)  # (B,), mean across upper-face vertices

    return torch.tensor(fdd) if not is_reduced else torch.tensor(fdd).nanmean(dim=0)


class TestUpperFaceDynamicsDeviation(MetricTester):
    """Test class for `UpperFaceDynamicsDeviation` metric (FDD)."""

    atol: float = 1e-2

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_fdd_metric_class(self, ddp):
        """Test class implementation of metric."""
        upper_face_map = [0, 1, 2, 3, 4]
        vertices_pred, vertices_gt, template = _generate_vertices()
        self.run_class_metric_test(
            ddp=ddp,
            preds=vertices_pred,
            target=vertices_gt,
            metric_class=UpperFaceDynamicsDeviation,
            reference_metric=partial(_reference_fdd, template=template, upper_face_map=upper_face_map, is_reduced=True),
            metric_args={"template": template, "upper_face_map": upper_face_map},
        )

    def test_fdd_functional(self):
        """Test functional implementation of metric."""
        upper_face_map = [0, 1, 2, 3, 4]
        vertices_pred, vertices_gt, template = _generate_vertices()

        self.run_functional_metric_test(
            preds=vertices_pred,
            target=vertices_gt,
            metric_functional=upper_face_dynamics_deviation,
            reference_metric=partial(_reference_fdd, template=template, upper_face_map=upper_face_map),
            metric_args={"template": template, "upper_face_map": upper_face_map},
        )

    def test_fdd_differentiability(self):
        """Test differentiability of FDD metric."""
        upper_face_map = [0, 1, 2, 3, 4]
        vertices_pred, vertices_gt, template = _generate_vertices()

        self.run_differentiability_test(
            preds=vertices_pred,
            target=vertices_gt,
            metric_module=UpperFaceDynamicsDeviation,
            metric_functional=upper_face_dynamics_deviation,
            metric_args={"template": template, "upper_face_map": upper_face_map},
        )

    def test_error_on_wrong_dimensions(self):
        """Test that an error is raised for wrong input dimensions."""
        metric = UpperFaceDynamicsDeviation(template=torch.randn(100, 3), upper_face_map=[0, 1, 2, 3, 4])
        with pytest.raises(
            ValueError, match="Expected both vertices_pred and vertices_gt to have 4 dimensions but got.*"
        ):
            metric(torch.randn(10, 100), torch.randn(10, 100, 3))

    def test_error_on_mismatched_dimensions(self):
        """Test that an error is raised for mismatched vertex dimensions."""
        metric = UpperFaceDynamicsDeviation(template=torch.randn(100, 3), upper_face_map=[0, 1, 2, 3, 4])
        with pytest.raises(
            ValueError,
            match="Expected vertices_pred and vertices_gt to have same vertex and coordinate dimensions but got.*",
        ):
            metric(torch.randn(10, 10, 80, 3), torch.randn(10, 10, 100, 3))

    def test_error_on_template_shape_mismatch(self):
        """Test that an error is raised when template shape does not match vertex-coordinate dimensions."""
        metric = UpperFaceDynamicsDeviation(template=torch.randn(100, 3), upper_face_map=[0, 1, 2, 3, 4])
        with pytest.raises(ValueError, match="Shape mismatch: expected template shape.*to match.*"):
            metric(torch.randn(10, 10, 120, 3), torch.randn(10, 10, 120, 3))

    def test_error_on_empty_upper_face_map(self):
        """Test that an error is raised if upper_face_map is empty."""
        with pytest.raises(ValueError, match="upper_face_map cannot be empty."):
            UpperFaceDynamicsDeviation(template=torch.randn(100, 3), upper_face_map=[])

    def test_error_on_invalid_upper_face_indices(self):
        """Test that an error is raised if upper_face_map has invalid indices."""
        with pytest.raises(ValueError, match="upper_face_map contains out-of-range vertex indices.*"):
            UpperFaceDynamicsDeviation(template=torch.randn(100, 3), upper_face_map=[98, 99, 100])

    def test_different_sequence_lengths(self):
        """Test that the metric handles different sequence lengths correctly."""
        metric = UpperFaceDynamicsDeviation(template=torch.randn(100, 3), upper_face_map=[0, 1, 2, 3, 4])
        metric(torch.randn(10, 10, 100, 3), torch.randn(10, 8, 100, 3))

    def test_plot_method(self):
        """Test the plot method of FDD."""
        vertices_pred, vertices_gt, template = _generate_vertices()
        metric = UpperFaceDynamicsDeviation(template=template, upper_face_map=[0, 1, 2, 3, 4])
        metric.update(vertices_pred[0], vertices_gt[0])
        fig, ax = metric.plot()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)
