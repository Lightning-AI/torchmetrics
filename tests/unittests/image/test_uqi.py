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
from skimage.metrics import structural_similarity
from torch import Tensor
from torchmetrics.functional.image.uqi import universal_image_quality_index
from torchmetrics.image.uqi import UniversalImageQualityIndex

from unittests import BATCH_SIZE, NUM_BATCHES
from unittests.helpers import seed_all
from unittests.helpers.testers import MetricTester

seed_all(42)


class _InputMultichannel(NamedTuple):
    preds: Tensor
    target: Tensor
    multichannel: bool


# UQI is SSIM with both constants k1 and k2 as 0
skimage_uqi = partial(structural_similarity, k1=0, k2=0)


_inputs = []
for size, channel, coef, multichannel, dtype in [
    (12, 3, 0.9, True, torch.float),
    (13, 1, 0.8, False, torch.float32),
    (14, 1, 0.7, False, torch.double),
    (15, 3, 0.6, True, torch.float64),
]:
    preds = torch.rand(NUM_BATCHES, BATCH_SIZE, channel, size, size, dtype=dtype)
    _inputs.append(
        _InputMultichannel(
            preds=preds,
            target=preds * coef,
            multichannel=multichannel,
        )
    )


def _skimage_uqi(preds, target, multichannel, kernel_size):
    c, h, w = preds.shape[-3:]
    sk_preds = preds.view(-1, c, h, w).permute(0, 2, 3, 1).numpy()
    sk_target = target.view(-1, c, h, w).permute(0, 2, 3, 1).numpy()
    if not multichannel:
        sk_preds = sk_preds[:, :, :, 0]
        sk_target = sk_target[:, :, :, 0]

    return skimage_uqi(
        sk_target,
        sk_preds,
        data_range=1.0,
        multichannel=multichannel,
        gaussian_weights=True,
        win_size=kernel_size,
        sigma=1.5,
        use_sample_covariance=False,
        channel_axis=-1,
    )


@pytest.mark.parametrize(
    "preds, target, multichannel",
    [(i.preds, i.target, i.multichannel) for i in _inputs],
)
@pytest.mark.parametrize("kernel_size", [5, 11])
class TestUQI(MetricTester):
    """Test class for `UniversalImageQualityIndex` metric."""

    atol = 6e-3

    @pytest.mark.parametrize("ddp", [True, False])
    def test_uqi(self, preds, target, multichannel, kernel_size, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            UniversalImageQualityIndex,
            partial(_skimage_uqi, multichannel=multichannel, kernel_size=kernel_size),
            metric_args={"kernel_size": (kernel_size, kernel_size)},
        )

    def test_uqi_functional(self, preds, target, multichannel, kernel_size):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            universal_image_quality_index,
            partial(_skimage_uqi, multichannel=multichannel, kernel_size=kernel_size),
            metric_args={"kernel_size": (kernel_size, kernel_size)},
        )

    # UQI half + cpu does not work due to missing support in torch.log
    @pytest.mark.xfail(reason="UQI metric does not support cpu + half precision")
    def test_uqi_half_cpu(self, preds, target, multichannel, kernel_size):
        """Test dtype support of the metric on CPU."""
        self.run_precision_test_cpu(
            preds,
            target,
            UniversalImageQualityIndex,
            universal_image_quality_index,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_uqi_half_gpu(self, preds, target, multichannel, kernel_size):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(
            preds,
            target,
            UniversalImageQualityIndex,
            universal_image_quality_index,
        )


@pytest.mark.parametrize(
    ("pred", "target", "kernel", "sigma", "match"),
    [
        ([1, 16, 16], [1, 16, 16], [11, 11], [1.5, 1.5], "Expected `preds` and `target` to have BxCxHxW shape.*"),
        (
            [1, 1, 16, 16],
            [1, 1, 16, 16],
            [11, 11],
            [1.5],
            "Expected `kernel_size` and `sigma` to have the length of two.*",
        ),
        (
            [1, 1, 16, 16],
            [1, 1, 16, 16],
            [11],
            [1.5, 1.5],
            "Expected `kernel_size` and `sigma` to have the length of two.*",
        ),
        ([1, 1, 16, 16], [1, 1, 16, 16], [11], [1.5], "Expected `kernel_size` and `sigma` to have the length of two.*"),
        ([1, 1, 16, 16], [1, 1, 16, 16], [11, 0], [1.5, 1.5], "Expected `kernel_size` to have odd positive number.*"),
        ([1, 1, 16, 16], [1, 1, 16, 16], [11, 10], [1.5, 1.5], "Expected `kernel_size` to have odd positive number.*"),
        ([1, 1, 16, 16], [1, 1, 16, 16], [11, -11], [1.5, 1.5], "Expected `kernel_size` to have odd positive number.*"),
        ([1, 1, 16, 16], [1, 1, 16, 16], [11, 11], [1.5, 0], "Expected `sigma` to have positive number.*"),
        ([1, 1, 16, 16], [1, 1, 16, 16], [11, 0], [1.5, -1.5], "Expected `kernel_size` to have odd positive number.*"),
    ],
)
def test_uqi_invalid_inputs(pred, target, kernel, sigma, match):
    """Check that errors are raised on wrong input and parameter combinations."""
    pred = torch.rand(pred)
    target = torch.rand(target)
    with pytest.raises(ValueError, match=match):
        universal_image_quality_index(pred, target, kernel, sigma)


def test_uqi_different_dtype():
    """Check that an type error is raised if preds and target have different dtype."""
    pred_t = torch.rand([1, 1, 16, 16])
    target_t = torch.rand([1, 1, 16, 16], dtype=torch.float64)
    with pytest.raises(TypeError, match="Expected `preds` and `target` to have the same data type.*"):
        universal_image_quality_index(pred_t, target_t)


def test_uqi_unequal_kernel_size():
    """Test the case where kernel_size[0] != kernel_size[1]."""
    preds = torch.tensor(
        [
            [
                [
                    [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                ]
            ]
        ]
    )
    target = torch.tensor(
        [
            [
                [
                    [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                ]
            ]
        ]
    )
    # kernel order matters
    torch.allclose(universal_image_quality_index(preds, target, kernel_size=(3, 5)), torch.tensor(0.10662283))
    torch.allclose(universal_image_quality_index(preds, target, kernel_size=(5, 3)), torch.tensor(0.10662283))
