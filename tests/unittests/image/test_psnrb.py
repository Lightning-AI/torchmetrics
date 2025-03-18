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
import math

import numpy as np
import pytest
import torch
from sewar.utils import _compute_bef

from torchmetrics.functional.image.psnrb import peak_signal_noise_ratio_with_blocked_effect
from torchmetrics.image import PeakSignalNoiseRatioWithBlockedEffect
from unittests import BATCH_SIZE, NUM_BATCHES
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

seed_all(42)

_input = (
    (torch.rand(NUM_BATCHES, BATCH_SIZE, 1, 16, 16), torch.rand(NUM_BATCHES, BATCH_SIZE, 1, 16, 16)),
    (
        torch.randint(0, 255, (NUM_BATCHES, BATCH_SIZE, 1, 16, 16)),
        torch.randint(0, 255, (NUM_BATCHES, BATCH_SIZE, 1, 16, 16)),
    ),
)


def _reference_psnrb(preds, target):
    """Reference implementation of PSNRB metric.

    Inspired by
    https://github.com/andrewekhalel/sewar/blob/master/sewar/full_ref.py
    that also supports batched inputs.

    """
    preds = preds.numpy()
    target = target.numpy()
    imdff = np.double(target) - np.double(preds)

    mse = np.mean(np.square(imdff.flatten()))
    bef = sum([_compute_bef(p.squeeze()) for p in preds])
    mse_b = mse + bef

    if np.amax(preds) > 2:
        psnr_b = 10 * math.log10((target.max() - target.min()) ** 2 / mse_b)
    else:
        psnr_b = 10 * math.log10(1 / mse_b)

    return psnr_b


@pytest.mark.parametrize("preds, target", _input)
class TestPSNR(MetricTester):
    """Test class for PSNRB metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_psnr(self, preds, target, ddp):
        """Test that modular PSNRB metric returns the same result as the reference implementation."""
        self.run_class_metric_test(
            ddp, preds, target, metric_class=PeakSignalNoiseRatioWithBlockedEffect, reference_metric=_reference_psnrb
        )

    def test_psnr_functional(self, preds, target):
        """Test that functional PSNRB metric returns the same result as the reference implementation."""
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=peak_signal_noise_ratio_with_blocked_effect,
            reference_metric=_reference_psnrb,
        )

    def test_psnr_half_cpu(self, preds, target):
        """Test that PSNRB metric works with half precision on cpu."""
        if target.max() - target.min() < 2:
            pytest.xfail("PSNRB metric does not support cpu + half precision")
        self.run_precision_test_cpu(
            preds,
            target,
            PeakSignalNoiseRatioWithBlockedEffect,
            peak_signal_noise_ratio_with_blocked_effect,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_psnr_half_gpu(self, preds, target):
        """Test that PSNRB metric works with half precision on gpu."""
        self.run_precision_test_gpu(
            preds,
            target,
            PeakSignalNoiseRatioWithBlockedEffect,
            peak_signal_noise_ratio_with_blocked_effect,
        )


def test_error_on_color_images():
    """Test that appropriate error is raised when color images are passed to PSNRB metric."""
    with pytest.raises(ValueError, match="`psnrb` metric expects grayscale images.*"):
        peak_signal_noise_ratio_with_blocked_effect(torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16))
