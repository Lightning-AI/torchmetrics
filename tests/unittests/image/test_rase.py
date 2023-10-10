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
from torchmetrics.functional.image.helper import _uniform_filter
from torchmetrics.image import RelativeAverageSpectralError

from unittests import BATCH_SIZE
from unittests.helpers.testers import MetricTester


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


def _sewar_rase(preds, target, window_size):
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


@pytest.mark.parametrize("preds, target, window_size", [(i.preds, i.target, i.window_size) for i in _inputs])
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
            RelativeAverageSpectralError,
            partial(_sewar_rase, window_size=window_size),
            metric_args={"window_size": window_size},
            check_batch=False,
        )

    def test_rase_functional(self, preds, target, window_size):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            relative_average_spectral_error,
            partial(_sewar_rase, window_size=window_size),
            metric_args={"window_size": window_size},
        )
