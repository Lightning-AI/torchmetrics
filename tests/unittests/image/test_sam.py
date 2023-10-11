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
from torch import Tensor
from torch.nn import functional as F  # noqa: N812
from torchmetrics.functional.image.sam import spectral_angle_mapper
from torchmetrics.image.sam import SpectralAngleMapper
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_2_1

from unittests import BATCH_SIZE, NUM_BATCHES, _Input
from unittests.helpers import seed_all
from unittests.helpers.testers import MetricTester

seed_all(42)


_inputs = []
for size, channel, dtype in [
    (12, 3, torch.float),
    (13, 3, torch.float32),
    (14, 3, torch.double),
    (15, 3, torch.float64),
]:
    preds = torch.rand(NUM_BATCHES, BATCH_SIZE, channel, size, size, dtype=dtype)
    target = torch.rand(NUM_BATCHES, BATCH_SIZE, channel, size, size, dtype=dtype)
    _inputs.append(_Input(preds=preds, target=target))


def _baseline_sam(
    preds: Tensor,
    target: Tensor,
    reduction: str = "elementwise_mean",
) -> Tensor:
    """Baseline implementation of spectral angle mapper."""
    reduction_options = ("elementwise_mean", "sum", "none")
    if reduction not in reduction_options:
        raise ValueError(f"reduction has to be one of {reduction_options}, got: {reduction}.")
    similarity = F.cosine_similarity(preds, target)
    sam_score = torch.clamp(similarity, -1, 1).acos()
    # reduction
    if reduction == "sum":
        return torch.sum(sam_score)
    if reduction == "elementwise_mean":
        return torch.mean(sam_score)
    return sam_score


@pytest.mark.parametrize("reduction", ["sum", "elementwise_mean"])
@pytest.mark.parametrize(
    "preds, target",
    [(i.preds, i.target) for i in _inputs],
)
class TestSpectralAngleMapper(MetricTester):
    """Test class for `SpectralAngleMapper` metric."""

    @pytest.mark.parametrize("ddp", [True, False])
    def test_sam(self, reduction, preds, target, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            SpectralAngleMapper,
            partial(_baseline_sam, reduction=reduction),
            metric_args={"reduction": reduction},
        )

    def test_sam_functional(self, reduction, preds, target):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            spectral_angle_mapper,
            partial(_baseline_sam, reduction=reduction),
            metric_args={"reduction": reduction},
        )

    # SAM half + cpu does not work due to missing support in torch.log
    @pytest.mark.skipif(
        not _TORCH_GREATER_EQUAL_2_1, reason="Pytoch below 2.1 does not support cpu + half precision used in SAM metric"
    )
    def test_sam_half_cpu(self, reduction, preds, target):
        """Test dtype support of the metric on CPU."""
        self.run_precision_test_cpu(
            preds,
            target,
            SpectralAngleMapper,
            spectral_angle_mapper,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_sam_half_gpu(self, reduction, preds, target):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(preds, target, SpectralAngleMapper, spectral_angle_mapper)


def test_error_on_different_shape(metric_class=SpectralAngleMapper):
    """Test that error is raised if preds and target have different shape."""
    metric = metric_class()
    with pytest.raises(RuntimeError, match="Predictions and targets are expected to have the same shape.*"):
        metric(torch.randn([1, 3, 16, 16]), torch.randn([1, 1, 16, 16]))


def test_error_on_invalid_shape(metric_class=SpectralAngleMapper):
    """Test that error is raised if input is not 4D."""
    metric = metric_class()
    with pytest.raises(ValueError, match="Expected `preds` and `target` to have BxCxHxW shape.*"):
        metric(torch.randn([3, 16, 16]), torch.randn([3, 16, 16]))


def test_error_on_invalid_type(metric_class=SpectralAngleMapper):
    """Test that error is raised if preds and target have different dtype."""
    metric = metric_class()
    with pytest.raises(TypeError, match="Expected `preds` and `target` to have the same data type.*"):
        metric(torch.randn([3, 16, 16]), torch.randn([3, 16, 16], dtype=torch.float64))


def test_error_on_grayscale_image(metric_class=SpectralAngleMapper):
    """Test that error is raised if number of channels is not larger than 1."""
    metric = metric_class()
    with pytest.raises(ValueError, match="Expected channel dimension of `preds` and `target` to be larger than 1.*"):
        metric(torch.randn([16, 1, 16, 16]), torch.randn([16, 1, 16, 16]))
