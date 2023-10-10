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

import numpy as np
import pytest
import torch
from torch import Tensor
from torchmetrics.functional.image.d_lambda import spectral_distortion_index
from torchmetrics.functional.image.uqi import universal_image_quality_index
from torchmetrics.image.d_lambda import SpectralDistortionIndex

from unittests import BATCH_SIZE, NUM_BATCHES
from unittests.helpers import seed_all
from unittests.helpers.testers import MetricTester

seed_all(42)


class _Input(NamedTuple):
    preds: Tensor
    target: Tensor
    p: int


_inputs = []
for size, channel, p, dtype in [
    (12, 3, 1, torch.float),
    (13, 1, 3, torch.float32),
    (14, 1, 4, torch.double),
    (15, 3, 1, torch.float64),
]:
    preds = torch.rand(NUM_BATCHES, BATCH_SIZE, channel, size, size, dtype=dtype)
    target = torch.rand(NUM_BATCHES, BATCH_SIZE, channel, size, size, dtype=dtype)
    _inputs.append(
        _Input(
            preds=preds,
            target=target,
            p=p,
        )
    )


def _baseline_d_lambda(preds: np.ndarray, target: np.ndarray, p: int = 1) -> float:
    """NumPy based implementation of Spectral Distortion Index, which uses UQI of TorchMetrics."""
    target, preds = torch.from_numpy(target), torch.from_numpy(preds)
    # Permute to ensure B x C x H x W (Pillow/NumPy stores in B x H x W x C)
    target = target.permute(0, 3, 1, 2)
    preds = preds.permute(0, 3, 1, 2)

    length = preds.shape[1]
    m1 = np.zeros((length, length), dtype=np.float32)
    m2 = np.zeros((length, length), dtype=np.float32)

    # Convert target and preds to Torch Tensors, pass them to metrics UQI
    # this is mainly because reference repo (sewar) uses uniform distribution
    # in their implementation of UQI, and we use gaussian distribution
    # and they have different default values for some kwargs like window size.
    for k in range(length):
        for r in range(k, length):
            m1[k, r] = m1[r, k] = universal_image_quality_index(target[:, k : k + 1, :, :], target[:, r : r + 1, :, :])
            m2[k, r] = m2[r, k] = universal_image_quality_index(preds[:, k : k + 1, :, :], preds[:, r : r + 1, :, :])
    diff = np.abs(m1 - m2) ** p

    # Special case: when number of channels (L) is 1, there will be only one element in M1 and M2. Hence no need to sum.
    if length == 1:
        return diff[0][0] ** (1.0 / p)
    return (1.0 / (length * (length - 1)) * np.sum(diff)) ** (1.0 / p)


def _np_d_lambda(preds, target, p):
    c, h, w = preds.shape[-3:]
    np_preds = preds.view(-1, c, h, w).permute(0, 2, 3, 1).numpy()
    np_target = target.view(-1, c, h, w).permute(0, 2, 3, 1).numpy()

    return _baseline_d_lambda(
        np_preds,
        np_target,
        p=p,
    )


@pytest.mark.parametrize(
    "preds, target, p",
    [(i.preds, i.target, i.p) for i in _inputs],
)
class TestSpectralDistortionIndex(MetricTester):
    """Test class for `SpectralDistortionIndex` metric."""

    atol = 6e-3

    @pytest.mark.parametrize("ddp", [True, False])
    def test_d_lambda(self, preds, target, p, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            SpectralDistortionIndex,
            partial(_np_d_lambda, p=p),
            metric_args={"p": p},
        )

    def test_d_lambda_functional(self, preds, target, p):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            spectral_distortion_index,
            partial(_np_d_lambda, p=p),
            metric_args={"p": p},
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_d_lambda_half_gpu(self, preds, target, p):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(preds, target, SpectralDistortionIndex, spectral_distortion_index, {"p": p})


@pytest.mark.parametrize(
    ("preds", "target", "p", "match"),
    [
        ([1, 16, 16], [1, 16, 16], 1, "Expected `preds` and `target` to have BxCxHxW shape.*"),  # len(shape)
        ([1, 1, 16, 16], [1, 1, 16, 16], 0, "Expected `p` to be a positive integer. Got p: 0."),  # invalid p
        ([1, 1, 16, 16], [1, 1, 16, 16], -1, "Expected `p` to be a positive integer. Got p: -1."),  # invalid p
    ],
)
def test_d_lambda_invalid_inputs(preds, target, p, match):
    """Test that invalid input raises the correct errors."""
    preds_t = torch.rand(preds)
    target_t = torch.rand(target)
    with pytest.raises(ValueError, match=match):
        spectral_distortion_index(preds_t, target_t, p)


def test_d_lambda_invalid_type():
    """Test that error is raised on different dtypes."""
    preds_t = torch.rand((1, 1, 16, 16))
    target_t = torch.rand((1, 1, 16, 16), dtype=torch.float64)
    with pytest.raises(TypeError, match="Expected `ms` and `fused` to have the same data type.*"):
        spectral_distortion_index(preds_t, target_t, p=1)


def test_d_lambda_different_sizes():
    """Since d lambda is reference free, it can accept different number of targets and preds."""
    preds = torch.rand(1, 1, 32, 32)
    target = torch.rand(1, 1, 16, 16)
    out = spectral_distortion_index(preds, target, p=1)
    assert isinstance(out, torch.Tensor)
