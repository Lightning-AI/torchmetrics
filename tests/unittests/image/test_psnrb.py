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
from functools import partial

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


def _reference_psnrb(preds, target, data_range):
    """Reference implementation of PSNRB metric.

    Inspired by
    https://github.com/andrewekhalel/sewar/blob/master/sewar/full_ref.py
    that also supports batched inputs.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        data_range: Range of the data. If not provided, it's determined from the data.

    Returns:
        PSNRB score

    """
    preds = preds.numpy()
    target = target.numpy()

    # Handle data_range parameter
    if isinstance(data_range, tuple):
        # Apply clamping if range is provided as tuple
        preds = np.clip(preds, data_range[0], data_range[1])
        target = np.clip(target, data_range[0], data_range[1])
        dr = data_range[1] - data_range[0]
    else:
        # Use the provided data_range directly
        dr = float(data_range)

    imdff = np.double(target) - np.double(preds)

    mse = np.mean(np.square(imdff.flatten()))
    bef = sum([_compute_bef(p.squeeze()) for p in preds])
    mse_b = mse + bef

    # Use the provided data_range for calculation
    return 10 * math.log10(dr**2 / mse_b)


@pytest.mark.parametrize(("preds", "target"), _input)
class TestPSNR(MetricTester):
    """Test class for PSNRB metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_psnr(self, preds, target, ddp):
        """Test that modular PSNRB metric returns the same result as the reference implementation."""
        # Pass a data_range value appropriate for the inputs
        data_range = 1.0 if preds.max() <= 1.0 else 255.0
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            metric_class=PeakSignalNoiseRatioWithBlockedEffect,
            reference_metric=partial(_reference_psnrb, data_range=data_range),
            metric_args={"data_range": data_range},
        )

    def test_psnr_functional(self, preds, target):
        """Test that functional PSNRB metric returns the same result as the reference implementation."""
        # Pass a data_range value appropriate for the inputs
        data_range = 1.0 if preds.max() <= 1.0 else 255.0
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=peak_signal_noise_ratio_with_blocked_effect,
            reference_metric=partial(_reference_psnrb, data_range=data_range),
            metric_args={"data_range": data_range},
        )

    def test_psnr_half_cpu(self, preds, target):
        """Test that PSNRB metric works with half precision on cpu."""
        if target.max() - target.min() < 2:
            pytest.xfail("PSNRB metric does not support cpu + half precision")
        # Pass a data_range value appropriate for the inputs
        data_range = 1.0 if preds.max() <= 1.0 else 255.0
        self.run_precision_test_cpu(
            preds,
            target,
            PeakSignalNoiseRatioWithBlockedEffect,
            peak_signal_noise_ratio_with_blocked_effect,
            metric_args={"data_range": data_range},
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_psnr_half_gpu(self, preds, target):
        """Test that PSNRB metric works with half precision on gpu."""
        # Pass a data_range value appropriate for the inputs
        data_range = 1.0 if preds.max() <= 1.0 else 255.0
        self.run_precision_test_gpu(
            preds,
            target,
            PeakSignalNoiseRatioWithBlockedEffect,
            peak_signal_noise_ratio_with_blocked_effect,
            metric_args={"data_range": data_range},
        )


def test_error_on_color_images():
    """Test that appropriate error is raised when color images are passed to PSNRB metric."""
    with pytest.raises(ValueError, match="`psnrb` metric expects grayscale images.*"):
        peak_signal_noise_ratio_with_blocked_effect(torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16), data_range=1.0)


class TestBlockSizeValidation:
    """Test that ``block_size`` validation rejects invalid values.

    Regression test for https://github.com/Lightning-AI/torchmetrics/issues/3364.
    The original ``and`` condition short-circuited on valid ``int`` types, so
    ``0``, ``-5``, and non-integer values slipped through silently or raised
    ``TypeError`` instead of the intended ``ValueError``.

    """

    @pytest.mark.parametrize("invalid_block_size", [0, -1, -5])
    def test_class_rejects_non_positive_int(self, invalid_block_size):
        """Class metric should raise ValueError for non-positive integers."""
        with pytest.raises(ValueError, match="Argument ``block_size`` should be a positive integer"):
            PeakSignalNoiseRatioWithBlockedEffect(data_range=1.0, block_size=invalid_block_size)

    @pytest.mark.parametrize("invalid_block_size", [1.5, 8.0])
    def test_class_rejects_float(self, invalid_block_size):
        """Class metric should raise ValueError for float values (even 8.0 is not int)."""
        with pytest.raises(ValueError, match="Argument ``block_size`` should be a positive integer"):
            PeakSignalNoiseRatioWithBlockedEffect(data_range=1.0, block_size=invalid_block_size)

    def test_class_rejects_string(self):
        """Class metric should raise ValueError for string values."""
        with pytest.raises(ValueError, match="Argument ``block_size`` should be a positive integer"):
            PeakSignalNoiseRatioWithBlockedEffect(data_range=1.0, block_size="foo")

    def test_class_accepts_valid_block_size(self):
        """Class metric should accept valid positive integer block sizes."""
        # Should not raise — just verify it constructs
        metric = PeakSignalNoiseRatioWithBlockedEffect(data_range=1.0, block_size=8)
        assert metric.block_size == 8
        metric = PeakSignalNoiseRatioWithBlockedEffect(data_range=1.0, block_size=1)
        assert metric.block_size == 1

    @pytest.mark.parametrize("invalid_block_size", [0, -1, -5])
    def test_functional_rejects_non_positive_int(self, invalid_block_size):
        """Functional metric should raise ValueError for non-positive integers."""
        preds = torch.rand(1, 1, 16, 16)
        target = torch.rand(1, 1, 16, 16)
        with pytest.raises(ValueError, match="Argument ``block_size`` should be a positive integer"):
            peak_signal_noise_ratio_with_blocked_effect(preds, target, data_range=1.0, block_size=invalid_block_size)

    @pytest.mark.parametrize("invalid_block_size", [1.5, 8.0])
    def test_functional_rejects_float(self, invalid_block_size):
        """Functional metric should raise ValueError for float values."""
        preds = torch.rand(1, 1, 16, 16)
        target = torch.rand(1, 1, 16, 16)
        with pytest.raises(ValueError, match="Argument ``block_size`` should be a positive integer"):
            peak_signal_noise_ratio_with_blocked_effect(preds, target, data_range=1.0, block_size=invalid_block_size)

    def test_functional_rejects_string(self):
        """Functional metric should raise ValueError for string values."""
        preds = torch.rand(1, 1, 16, 16)
        target = torch.rand(1, 1, 16, 16)
        with pytest.raises(ValueError, match="Argument ``block_size`` should be a positive integer"):
            peak_signal_noise_ratio_with_blocked_effect(preds, target, data_range=1.0, block_size="foo")

    def test_functional_accepts_valid_block_size(self):
        """Functional metric should work with valid positive integer block sizes."""
        preds = torch.rand(1, 1, 16, 16)
        target = torch.rand(1, 1, 16, 16)
        # Should not raise
        result = peak_signal_noise_ratio_with_blocked_effect(preds, target, data_range=1.0, block_size=8)
        assert torch.isfinite(result)
