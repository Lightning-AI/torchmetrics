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
from weakref import ref

import pytest
import torch
from torch import Tensor

from torchmetrics.functional.image.ergas import error_relative_global_dimensionless_synthesis
from torchmetrics.image.ergas import ErrorRelativeGlobalDimensionlessSynthesis
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
    ("preds", "target", "ratio"),
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


@pytest.mark.parametrize("reduction", ["elementwise_mean", "sum", "none", None])
def test_ergas_updates_with_different_image_sizes(reduction):
    """Accumulate per-image scores across differently sized batches."""
    metric = ErrorRelativeGlobalDimensionlessSynthesis(reduction=reduction)
    scores = []
    for shape, scale in [((2, 3, 4, 5), 0.5), ((3, 3, 7, 2), 0.8)]:
        preds = torch.arange(1, 1 + torch.tensor(shape).prod().item(), dtype=torch.float64).reshape(shape)
        target = preds * scale
        scores.append(error_relative_global_dimensionless_synthesis(preds, target, reduction="none"))
        metric.update(preds, target)

    expected = torch.cat(scores)
    if reduction == "elementwise_mean":
        expected = expected.mean()
    elif reduction == "sum":
        expected = expected.sum()
    torch.testing.assert_close(metric.compute(), expected)


@pytest.mark.parametrize("reduction", ["elementwise_mean", "sum", "none", None])
def test_ergas_does_not_retain_input_images(reduction):
    """Updates should not keep full images in metric state."""
    metric = ErrorRelativeGlobalDimensionlessSynthesis(reduction=reduction)
    preds = torch.rand(2, 3, 8, 8)
    target = torch.rand_like(preds)
    preds_ref, target_ref = ref(preds), ref(target)
    metric.update(preds, target)
    del preds, target
    assert preds_ref() is None
    assert target_ref() is None


@pytest.mark.parametrize("reduction", ["elementwise_mean", "sum", "none"])
def test_ergas_forward_keeps_gradients(reduction):
    """The streaming state must preserve gradients for a batch's forward pass."""
    preds = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]], [[[2.0, 3.0], [4.0, 5.0]]]], requires_grad=True)
    target = torch.tensor([[[[1.5, 2.5], [2.0, 3.0]]], [[[1.0, 2.0], [3.0, 4.0]]]])
    metric = ErrorRelativeGlobalDimensionlessSynthesis(reduction=reduction)

    actual = metric(preds, target)
    expected = error_relative_global_dimensionless_synthesis(preds, target, reduction=reduction)
    actual_grad = torch.autograd.grad(actual.sum(), preds, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.sum(), preds)[0]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("reduction", ["elementwise_mean", "sum", "none", None])
def test_ergas_merge_states(reduction):
    """Combine uneven batches without weighting each batch equally."""
    first = ErrorRelativeGlobalDimensionlessSynthesis(reduction=reduction)
    second = ErrorRelativeGlobalDimensionlessSynthesis(reduction=reduction)
    preds = torch.arange(1.0, 25.0).reshape(3, 1, 2, 4)
    target = preds * torch.tensor([0.5, 0.8, 0.9]).reshape(3, 1, 1, 1)
    first.update(preds[:1], target[:1])
    second.update(preds[1:], target[1:])

    first.merge_state(second)
    expected = error_relative_global_dimensionless_synthesis(preds, target, reduction=reduction)
    actual = first.compute()
    if reduction in ("none", None):
        actual = actual.sort().values
        expected = expected.sort().values
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("reduction", "expected"), [("elementwise_mean", 75.0), ("sum", 76800.0)])
def test_ergas_half_precision_large_batch(reduction, expected):
    """Accumulating many half-precision scores should not overflow their sum."""
    preds = torch.ones(1024, 1, 1, 1, dtype=torch.float16)
    target = preds * 0.25
    metric = ErrorRelativeGlobalDimensionlessSynthesis(reduction=reduction)
    metric.update(preds, target)
    torch.testing.assert_close(metric.compute(), torch.tensor(expected))
