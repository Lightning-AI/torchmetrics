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

import pytest
import torch
from torch import Tensor

from torchmetrics.functional.image.ergas import error_relative_global_dimensionless_synthesis
from torchmetrics.image.ergas import ErrorRelativeGlobalDimensionlessSynthesis
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_2_1
from unittests import BATCH_SIZE, NUM_BATCHES
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

seed_all(42)


class _Input(NamedTuple):
    preds: Tensor
    target: Tensor
    ratio: int


_inputs = []
for size, channel, coef, ratio, dtype in [
    (12, 1, 0.9, 1, torch.float),
    (13, 3, 0.8, 2, torch.float32),
    (14, 1, 0.7, 3, torch.double),
    (15, 3, 0.5, 4, torch.float64),
]:
    preds = torch.rand(NUM_BATCHES, BATCH_SIZE, channel, size, size, dtype=dtype)
    _inputs.append(_Input(preds=preds, target=preds * coef, ratio=ratio))


def _reference_ergas(
    preds: Tensor,
    target: Tensor,
    ratio: float = 4,
    reduction: str = "elementwise_mean",
) -> Tensor:
    """Baseline implementation of Erreur Relative Globale Adimensionnelle de Synthèse."""
    reduction_options = ("elementwise_mean", "sum", "none")
    if reduction not in reduction_options:
        raise ValueError(f"reduction has to be one of {reduction_options}, got: {reduction}.")
    # reshape to (batch_size, channel, height*width)
    b, c, h, w = preds.shape
    sk_preds = preds.reshape(b, c, h * w)
    sk_target = target.reshape(b, c, h * w)
    # compute rmse per band
    diff = sk_preds - sk_target
    sum_squared_error = torch.sum(diff * diff, dim=2)
    rmse_per_band = torch.sqrt(sum_squared_error / (h * w))
    mean_target = torch.mean(sk_target, dim=2)
    # compute ergas score
    ergas_score = 100 / ratio * torch.sqrt(torch.sum((rmse_per_band / mean_target) ** 2, dim=1) / c)
    # reduction
    if reduction == "sum":
        return torch.sum(ergas_score)
    if reduction == "elementwise_mean":
        return torch.mean(ergas_score)
    return ergas_score


@pytest.mark.parametrize("reduction", ["sum", "elementwise_mean"])
@pytest.mark.parametrize(
    "preds, target, ratio",
    [(i.preds, i.target, i.ratio) for i in _inputs],
)
class TestErrorRelativeGlobalDimensionlessSynthesis(MetricTester):
    """Test class for `ErrorRelativeGlobalDimensionlessSynthesis` metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_ergas(self, reduction, preds, target, ratio, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            metric_class=ErrorRelativeGlobalDimensionlessSynthesis,
            reference_metric=partial(_reference_ergas, ratio=ratio, reduction=reduction),
            metric_args={"ratio": ratio, "reduction": reduction},
        )

    def test_ergas_functional(self, reduction, preds, target, ratio):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=error_relative_global_dimensionless_synthesis,
            reference_metric=partial(_reference_ergas, ratio=ratio, reduction=reduction),
            metric_args={"ratio": ratio, "reduction": reduction},
        )

    # ERGAS half + cpu does not work due to missing support in torch.log
    @pytest.mark.skipif(
        not _TORCH_GREATER_EQUAL_2_1,
        reason="Pytoch below 2.1 does not support cpu + half precision used in ERGAS metric",
    )
    def test_ergas_half_cpu(self, reduction, preds, target, ratio):
        """Test dtype support of the metric on CPU."""
        self.run_precision_test_cpu(
            preds,
            target,
            ErrorRelativeGlobalDimensionlessSynthesis,
            error_relative_global_dimensionless_synthesis,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_ergas_half_gpu(self, reduction, preds, target, ratio):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(
            preds, target, ErrorRelativeGlobalDimensionlessSynthesis, error_relative_global_dimensionless_synthesis
        )


def test_error_on_different_shape(metric_class=ErrorRelativeGlobalDimensionlessSynthesis):
    """Check that error is raised when input have different shape."""
    metric = metric_class()
    with pytest.raises(RuntimeError, match="Predictions and targets are expected to have the same shape.*"):
        metric(torch.randn([1, 3, 16, 16]), torch.randn([1, 1, 16, 16]))


def test_error_on_invalid_shape(metric_class=ErrorRelativeGlobalDimensionlessSynthesis):
    """Check that error is raised when input is not 4D."""
    metric = metric_class()
    with pytest.raises(ValueError, match="Expected `preds` and `target` to have BxCxHxW shape.*"):
        metric(torch.randn([3, 16, 16]), torch.randn([3, 16, 16]))


def test_error_on_invalid_type(metric_class=ErrorRelativeGlobalDimensionlessSynthesis):
    """Test that error is raised if preds and target have different dtype."""
    metric = metric_class()
    with pytest.raises(TypeError, match="Expected `preds` and `target` to have the same data type.*"):
        metric(torch.randn([3, 16, 16]), torch.randn([3, 16, 16], dtype=torch.float64))
