# Copyright The PyTorch Lightning team.
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
import sewar
import torch
from torch import Tensor

from torchmetrics.functional import relative_average_spectral_error
from torchmetrics.functional.image.utils import _uniform_filter
from torchmetrics.image import RelativeAverageSpectralError
from unittests import BATCH_SIZE
from unittests._helpers.testers import MetricTester


class _InputWindowSized(NamedTuple):
    preds: Tensor
    target: Tensor
    window_size: int


_inputs = []
for size, channel, window_size, dtype in [
    (12, 3, 3, torch.float),
    (13, 1, 4, torch.float32),
    (14, 1, 5, torch.double),
    (15, 3, 8, torch.float64),
]:
    preds = torch.rand(2, BATCH_SIZE, channel, size, size, dtype=dtype)
    target = torch.rand(2, BATCH_SIZE, channel, size, size, dtype=dtype)
    _inputs.append(_InputWindowSized(preds=preds, target=target, window_size=window_size))


def _reference_sewar_rase(preds, target, window_size):
    """Baseline implementation of metric.

    This custom implementation is necessary since sewar only supports single image and aggregation therefore needs
    adjustments.

    """
    target_sum = torch.sum(_uniform_filter(target, window_size) / (window_size**2), dim=0)
    target_mean = target_sum / target.shape[0]
    target_mean = target_mean.mean(0)  # mean over image channels

    preds = preds.permute(0, 2, 3, 1).numpy()
    target = target.permute(0, 2, 3, 1).numpy()

    rmse_mean = torch.zeros(*preds.shape[1:])
    for pred, tgt in zip(preds, target):
        _, rmse_map = sewar.rmse_sw(tgt, pred, window_size)
        rmse_mean += rmse_map
    rmse_mean /= preds.shape[0]

    rase_map = 100 / target_mean * torch.sqrt(torch.mean(rmse_mean**2, 2))
    crop_slide = round(window_size / 2)

    return torch.mean(rase_map[crop_slide:-crop_slide, crop_slide:-crop_slide])


@pytest.mark.parametrize(("preds", "target", "window_size"), [(i.preds, i.target, i.window_size) for i in _inputs])
class TestRelativeAverageSpectralError(MetricTester):
    """Testing of Relative Average Spectral Error."""

    atol = 1e-2

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_rase(self, preds, target, window_size, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            metric_class=RelativeAverageSpectralError,
            reference_metric=partial(_reference_sewar_rase, window_size=window_size),
            metric_args={"window_size": window_size},
            check_batch=False,
        )

    def test_rase_functional(self, preds, target, window_size):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=relative_average_spectral_error,
            reference_metric=partial(_reference_sewar_rase, window_size=window_size),
            metric_args={"window_size": window_size},
        )


def test_rase_does_not_retain_inputs_and_state_is_bounded():
    """The accumulated state should be independent of the number of images."""
    metric = RelativeAverageSpectralError(window_size=3)
    sizes = []
    for batch_size in (1, 2, 3):
        preds = torch.rand(batch_size, 2, 8, 8)
        target = torch.rand_like(preds)
        preds_ref, target_ref = ref(preds), ref(target)
        metric.update(preds, target)
        del preds, target
        assert preds_ref() is None
        assert target_ref() is None
        sizes.append(sum(x.numel() for x in metric.metric_state.values() if isinstance(x, Tensor)))
    assert sizes[0] == sizes[1] == sizes[2]


def test_rase_uneven_updates_and_merge():
    """Accumulate map sums, rather than averaging the per-batch RASE values."""
    preds = torch.rand(5, 2, 8, 8, dtype=torch.float64)
    target = torch.rand_like(preds) + 0.1
    metric = RelativeAverageSpectralError(window_size=3)
    other = RelativeAverageSpectralError(window_size=3)
    metric.update(preds[:1], target[:1])
    other.update(preds[1:3], target[1:3])
    other.update(preds[3:], target[3:])
    metric.merge_state(other)
    expected = relative_average_spectral_error(preds, target, window_size=3)
    torch.testing.assert_close(metric.compute(), expected)


def test_rase_synchronizes_and_restores_local_state():
    """Synchronize unequal partitions before computing the nonlinear result."""
    preds = torch.rand(5, 2, 8, 8, dtype=torch.float64)
    target = torch.rand_like(preds) + 0.1
    local = RelativeAverageSpectralError(window_size=3)
    remote = RelativeAverageSpectralError(window_size=3)
    local.update(preds[:2], target[:2])
    remote.update(preds[2:], target[2:])
    saved = {name: value.clone() for name, value in local.metric_state.items()}
    remote_values = {id(value): remote.metric_state[name] for name, value in local.metric_state.items()}

    def gather(value, group=None):
        return [value, remote_values[id(value)]]

    local.dist_sync_fn = gather
    local.distributed_available_fn = lambda: True
    actual = local.compute()
    expected = relative_average_spectral_error(preds, target, window_size=3)
    torch.testing.assert_close(actual, expected)
    for name, value in local.metric_state.items():
        torch.testing.assert_close(value, saved[name])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_rase_forward_preserves_gradients(dtype):
    """The batch value and gradient should match the functional API."""
    preds = torch.rand(2, 2, 8, 8, dtype=dtype, requires_grad=True)
    target = torch.rand_like(preds) + 0.1
    actual = RelativeAverageSpectralError(window_size=3)(preds, target)
    ref_preds = preds.float() if dtype in (torch.float16, torch.bfloat16) else preds
    ref_target = target.float() if dtype in (torch.float16, torch.bfloat16) else target
    expected = relative_average_spectral_error(ref_preds, ref_target, window_size=3)
    actual_grad = torch.autograd.grad(actual, preds, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, preds)[0]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, expected_grad)


def test_rase_reset_allows_new_image_shape():
    """After reset, a new set of images can use different dimensions."""
    metric = RelativeAverageSpectralError(window_size=3)
    metric.update(torch.rand(2, 2, 8, 8), torch.rand(2, 2, 8, 8))
    metric.reset()
    preds = torch.rand(1, 3, 10, 9)
    target = torch.rand_like(preds) + 0.1
    metric.update(preds, target)
    torch.testing.assert_close(metric.compute(), relative_average_spectral_error(preds, target, window_size=3))


@pytest.mark.parametrize("use_forward", [False, True])
def test_rase_rejects_different_image_shapes(use_forward):
    """Do not silently broadcast differently shaped images between updates."""
    metric = RelativeAverageSpectralError(window_size=3)
    metric.update(torch.rand(1, 2, 8, 8), torch.rand(1, 2, 8, 8))
    saved = {name: value.clone() for name, value in metric.metric_state.items()}
    with pytest.raises(ValueError, match="same shape across updates"):
        (metric if use_forward else metric.update)(torch.rand(1, 1, 8, 8), torch.rand(1, 1, 8, 8))
    for name, value in metric.metric_state.items():
        torch.testing.assert_close(value, saved[name])


def test_rase_rejects_merging_different_image_shapes():
    """Merging metric states must not broadcast incompatible image maps."""
    first = RelativeAverageSpectralError(window_size=3)
    second = RelativeAverageSpectralError(window_size=3)
    first.update(torch.rand(1, 2, 8, 8), torch.rand(1, 2, 8, 8))
    second.update(torch.rand(1, 1, 8, 8), torch.rand(1, 1, 8, 8))
    saved = {name: value.clone() for name, value in first.metric_state.items()}
    with pytest.raises(ValueError, match="same shape across updates"):
        first.merge_state(second)
    for name, value in first.metric_state.items():
        torch.testing.assert_close(value, saved[name])


def test_rase_rejects_different_input_dtypes():
    """Low-precision promotion must not hide a mismatch between inputs."""
    metric = RelativeAverageSpectralError(window_size=3)
    with pytest.raises(TypeError, match="same data type"):
        metric.update(torch.rand(1, 1, 8, 8, dtype=torch.float16), torch.rand(1, 1, 8, 8))


@pytest.mark.parametrize("dtypes", [(torch.float32, torch.float64), (torch.float64, torch.float32)])
def test_rase_promotes_dtype_across_updates(dtypes):
    """Accumulated results retain the promoted dtype when input precision changes."""
    preds = [torch.rand(2, 2, 8, 8, dtype=dtype) for dtype in dtypes]
    target = [torch.rand_like(batch) + 0.1 for batch in preds]
    metric = RelativeAverageSpectralError(window_size=3)
    for pred, tgt in zip(preds, target):
        metric.update(pred, tgt)
    expected = relative_average_spectral_error(torch.cat(preds), torch.cat(target), window_size=3)
    torch.testing.assert_close(metric.compute(), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rase_low_precision_accumulation(dtype):
    """Repeated updates should not lose small increments in reduced precision."""
    metric = RelativeAverageSpectralError(window_size=3)
    preds = torch.full((2048, 1, 8, 8), 0.5, dtype=dtype)
    target = torch.ones_like(preds)
    metric.update(preds, target)
    for _ in range(8):
        metric.update(preds[:7], target[:7])
    expected = relative_average_spectral_error(preds[:1].float(), target[:1].float(), window_size=3)
    torch.testing.assert_close(metric.compute(), expected, rtol=5e-4, atol=5e-3)
