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
    target_sum = torch.sum(_uniform_filter(target, window_size), dim=0)
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

    @pytest.mark.parametrize("ddp", [False])
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


@pytest.mark.parametrize("window_size", [3, 4, 8])
def test_rase_single_image_matches_sewar(window_size):
    """For a single image the result should match ``sewar.rase`` directly."""
    preds = torch.rand(1, 3, 16, 16, dtype=torch.float64)
    target = torch.rand(1, 3, 16, 16, dtype=torch.float64)
    expected = sewar.rase(target[0].permute(1, 2, 0).numpy(), preds[0].permute(1, 2, 0).numpy(), window_size)

    result = relative_average_spectral_error(preds, target, window_size=window_size)
    assert torch.allclose(result, torch.tensor(expected, dtype=result.dtype))

    metric = RelativeAverageSpectralError(window_size=window_size)
    metric.update(preds, target)
    assert torch.allclose(metric.compute(), torch.tensor(expected, dtype=result.dtype))


def test_rase_constant_offset():
    """A constant error of 0.01 on an image of constant value 0.5 is a relative error of 2%."""
    target = torch.full((2, 3, 16, 16), 0.5, dtype=torch.float64)
    preds = target + 0.01
    result = relative_average_spectral_error(preds, target, window_size=8)
    assert torch.allclose(result, torch.tensor(2.0, dtype=torch.float64))
