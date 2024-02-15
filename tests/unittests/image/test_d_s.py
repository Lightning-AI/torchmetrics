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
from typing import Dict, List, NamedTuple

import numpy as np
import pytest
import torch
from scipy.ndimage import uniform_filter
from skimage.transform import resize
from torch import Tensor
from torchmetrics.functional.image.d_s import spatial_distortion_index
from torchmetrics.functional.image.uqi import universal_image_quality_index
from torchmetrics.image.d_s import SpatialDistortionIndex

from unittests import BATCH_SIZE, NUM_BATCHES
from unittests.helpers import seed_all
from unittests.helpers.testers import MetricTester

seed_all(42)


class _Input(NamedTuple):
    preds: Tensor
    target: List[Dict[str, Tensor]]
    ms: Tensor
    pan: Tensor
    pan_lr: Tensor
    norm_order: int
    window_size: int


_inputs = []
for size, channel, norm_order, r, window_size, pan_lr_exists, dtype in [
    (12, 3, 1, 16, 3, False, torch.float),
    (13, 1, 3, 8, 5, False, torch.float32),
    (14, 1, 4, 4, 5, True, torch.double),
    (15, 3, 1, 2, 3, True, torch.float64),
]:
    preds = torch.rand(NUM_BATCHES, BATCH_SIZE, channel, size * r, size * r, dtype=dtype)
    ms = torch.rand(NUM_BATCHES, BATCH_SIZE, channel, size, size, dtype=dtype)
    pan = torch.rand(NUM_BATCHES, BATCH_SIZE, channel, size * r, size * r, dtype=dtype)
    pan_lr = torch.rand(NUM_BATCHES, BATCH_SIZE, channel, size, size, dtype=dtype)
    _inputs.append(
        _Input(
            preds=preds,
            target=[
                {
                    "ms": ms[i],
                    "pan": pan[i],
                    **({"pan_lr": pan_lr[i]} if pan_lr_exists else {}),
                }
                for i in range(NUM_BATCHES)
            ],
            ms=ms,
            pan=pan,
            pan_lr=pan_lr if pan_lr_exists else None,
            norm_order=norm_order,
            window_size=window_size,
        )
    )


def _baseline_d_s(
    preds: np.ndarray,
    ms: np.ndarray,
    pan: np.ndarray,
    pan_lr: np.ndarray = None,
    norm_order: int = 1,
    window_size: int = 7,
) -> float:
    """NumPy based implementation of Spatial Distortion Index, which uses UQI of TorchMetrics."""
    pan_degraded = pan_lr
    if pan_degraded is None:
        try:
            pan_degraded = uniform_filter(pan, size=window_size, axes=[1, 2])
        except TypeError:
            pan_degraded = np.array([
                [uniform_filter(pan[i, ..., j], size=window_size) for j in range(pan.shape[-1])]
                for i in range(len(pan))
            ]).transpose((0, 2, 3, 1))
        pan_degraded = np.array([resize(img, ms.shape[1:3], anti_aliasing=False) for img in pan_degraded])

    length = preds.shape[-1]
    m1 = np.zeros(length, dtype=np.float32)
    m2 = np.zeros(length, dtype=np.float32)

    # Convert target and preds to Torch Tensors, pass them to metrics UQI
    # this is mainly because reference repo (sewar) uses uniform distribution
    # in their implementation of UQI, and we use gaussian distribution
    # and they have different default values for some kwargs like window size.
    ms = torch.from_numpy(ms).permute(0, 3, 1, 2)
    pan = torch.from_numpy(pan).permute(0, 3, 1, 2)
    preds = torch.from_numpy(preds).permute(0, 3, 1, 2)
    pan_degraded = torch.from_numpy(pan_degraded).permute(0, 3, 1, 2)
    for i in range(length):
        m1[i] = universal_image_quality_index(ms[:, i : i + 1], pan_degraded[:, i : i + 1])
        m2[i] = universal_image_quality_index(preds[:, i : i + 1], pan[:, i : i + 1])
    diff = np.abs(m1 - m2) ** norm_order
    return np.mean(diff) ** (1 / norm_order)


def _np_d_s(preds, target, pan=None, pan_lr=None, norm_order=1, window_size=7):
    np_preds = preds.permute(0, 2, 3, 1).cpu().numpy()
    if isinstance(target, dict):
        assert "ms" in target, "Expected `target` to contain 'ms'."
        np_ms = target["ms"].permute(0, 2, 3, 1).cpu().numpy()
        assert "pan" in target, "Expected `target` to contain 'pan'."
        np_pan = target["pan"].permute(0, 2, 3, 1).cpu().numpy()
        np_pan_lr = target["pan_lr"].permute(0, 2, 3, 1).cpu().numpy() if "pan_lr" in target else None
    else:
        np_ms = target.permute(0, 2, 3, 1).cpu().numpy()
        np_pan = pan.permute(0, 2, 3, 1).cpu().numpy()
        np_pan_lr = pan_lr.permute(0, 2, 3, 1).cpu().numpy() if pan_lr is not None else None

    return _baseline_d_s(
        np_preds,
        np_ms,
        np_pan,
        np_pan_lr,
        norm_order=norm_order,
        window_size=window_size,
    )


def _invoke_spatial_distortion_index(preds, target, ms, pan, pan_lr, norm_order, window_size):
    ms = target.get("ms", ms)
    pan = target.get("pan", pan)
    pan_lr = target.get("pan_lr", pan_lr)
    return spatial_distortion_index(preds, ms, pan, pan_lr, norm_order, window_size)


@pytest.mark.parametrize(
    "preds, target, ms, pan, pan_lr, norm_order, window_size",
    [(i.preds, i.target, i.ms, i.pan, i.pan_lr, i.norm_order, i.window_size) for i in _inputs],
)
class TestSpatialDistortionIndex(MetricTester):
    """Test class for `SpatialDistortionIndex` metric."""

    atol = 3e-6

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_d_s(self, preds, target, ms, pan, pan_lr, norm_order, window_size, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            SpatialDistortionIndex,
            partial(_np_d_s, norm_order=norm_order, window_size=window_size),
            metric_args={"norm_order": norm_order, "window_size": window_size},
        )

    def test_d_s_functional(self, preds, target, ms, pan, pan_lr, norm_order, window_size):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            ms,
            spatial_distortion_index,
            partial(_np_d_s, norm_order=norm_order, window_size=window_size),
            metric_args={"norm_order": norm_order, "window_size": window_size},
            fragment_kwargs=True,
            pan=pan,
            pan_lr=pan_lr,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_d_s_half_gpu(self, preds, target, ms, pan, pan_lr, norm_order, window_size):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(
            preds,
            target,
            SpatialDistortionIndex,
            partial(
                _invoke_spatial_distortion_index,
                ms=ms,
                pan=pan,
                pan_lr=pan_lr,
                norm_order=norm_order,
                window_size=window_size,
            ),
            {"norm_order": norm_order, "window_size": window_size},
        )


@pytest.mark.parametrize(
    ("preds", "ms", "pan", "pan_lr", "norm_order", "window_size", "match"),
    [
        (
            [1, 16, 16],
            [1, 1, 4, 4],
            [1, 1, 16, 16],
            None,
            1,
            3,
            "Expected `preds` to have BxCxHxW shape.*",
        ),  # len(preds.shape)
        (
            [1, 1, 16, 16],
            [1, 4, 4],
            [1, 1, 16, 16],
            None,
            1,
            3,
            "Expected `ms` to have BxCxHxW shape.*",
        ),  # len(target.shape)
        (
            [1, 1, 16, 16],
            [1, 1, 4, 4],
            [1, 16, 16],
            None,
            1,
            3,
            "Expected `pan` to have BxCxHxW shape.*",
        ),  # len(target.shape)
        (
            [1, 1, 16, 16],
            [1, 1, 4, 4],
            [1, 1, 16, 16],
            [1, 4, 4],
            1,
            3,
            "Expected `pan_lr` to have BxCxHxW shape.*",
        ),  # len(target.shape)
        (
            [1, 1, 16, 16],
            [1, 1, 4, 4],
            [1, 1, 16, 16],
            None,
            0,
            3,
            "Expected `norm_order` to be a positive integer. Got norm_order: 0.",
        ),  # invalid p
        (
            [1, 1, 16, 16],
            [1, 1, 4, 4],
            [1, 1, 16, 16],
            None,
            -1,
            3,
            "Expected `norm_order` to be a positive integer. Got norm_order: -1.",
        ),  # invalid p
        (
            [1, 1, 16, 16],
            [1, 1, 4, 4],
            [1, 1, 16, 16],
            None,
            1,
            0,
            "Expected `window_size` to be a positive integer. Got window_size: 0.",
        ),  # invalid window_size
        (
            [1, 1, 16, 16],
            [1, 1, 4, 4],
            [1, 1, 16, 16],
            None,
            1,
            -1,
            "Expected `window_size` to be a positive integer. Got window_size: -1.",
        ),  # invalid window_size
        (
            [1, 1, 16, 16],
            [1, 1, 4, 4],
            [1, 1, 17, 16],
            None,
            1,
            3,
            "Expected `preds` and `pan` to have the same height.*",
        ),  # invalid pan_h
        (
            [1, 1, 16, 16],
            [1, 1, 4, 4],
            [1, 1, 16, 17],
            None,
            1,
            3,
            "Expected `preds` and `pan` to have the same width.*",
        ),  # invalid pan_w
        (
            [1, 1, 16, 16],
            [1, 1, 5, 4],
            [1, 1, 16, 16],
            None,
            1,
            3,
            "Expected height of `preds` to be multiple of height of `ms`.*",
        ),  # invalid ms_h
        (
            [1, 1, 16, 16],
            [1, 1, 4, 5],
            [1, 1, 16, 16],
            None,
            1,
            3,
            "Expected width of `preds` to be multiple of width of `ms`.*",
        ),  # invalid ms_w
        (
            [1, 1, 16, 16],
            [1, 1, 4, 4],
            [1, 1, 16, 16],
            [1, 1, 5, 4],
            1,
            3,
            "Expected `ms` and `pan_lr` to have the same height.*",
        ),  # invalid pan_lr_h
        (
            [1, 1, 16, 16],
            [1, 1, 4, 4],
            [1, 1, 16, 16],
            [1, 1, 4, 5],
            1,
            3,
            "Expected `ms` and `pan_lr` to have the same width.*",
        ),  # invalid pan_lr_w
        (
            [1, 1, 16, 16],
            [1, 2, 4, 4],
            [1, 1, 16, 16],
            None,
            1,
            3,
            "Expected `preds` and `ms` to have the same batch and channel.*",
        ),  # invalid ms.shape
        (
            [1, 1, 16, 16],
            [2, 1, 4, 4],
            [1, 1, 16, 16],
            None,
            1,
            3,
            "Expected `preds` and `ms` to have the same batch and channel.*",
        ),  # invalid ms.shape
        (
            [1, 1, 16, 16],
            [1, 1, 4, 4],
            [1, 2, 16, 16],
            None,
            1,
            3,
            "Expected `preds` and `pan` to have the same batch and channel.*",
        ),  # invalid pan.shape
        (
            [1, 1, 16, 16],
            [1, 1, 4, 4],
            [2, 1, 16, 16],
            None,
            1,
            3,
            "Expected `preds` and `pan` to have the same batch and channel.*",
        ),  # invalid pan.shape
        (
            [1, 1, 16, 16],
            [1, 1, 4, 4],
            [1, 1, 16, 16],
            [1, 2, 4, 4],
            1,
            3,
            "Expected `preds` and `pan_lr` to have the same batch and channel.*",
        ),  # invalid pan_lr.shape
        (
            [1, 1, 16, 16],
            [1, 1, 4, 4],
            [1, 1, 16, 16],
            [2, 1, 4, 4],
            1,
            3,
            "Expected `preds` and `pan_lr` to have the same batch and channel.*",
        ),  # invalid pan_lr.shape
        (
            [1, 1, 16, 16],
            [1, 1, 4, 4],
            [1, 1, 16, 16],
            None,
            1,
            5,
            "Expected `window_size` to be smaller than dimension of `ms`.*",
        ),  # invalid window_size
    ],
)
def test_d_s_invalid_inputs(preds, ms, pan, pan_lr, norm_order, window_size, match):
    """Test that invalid input raises the correct errors."""
    preds_t = torch.rand(preds)
    ms_t = torch.rand(ms)
    pan_t = torch.rand(pan)
    pan_lr_t = torch.rand(pan_lr) if pan_lr is not None else None
    with pytest.raises(ValueError, match=match):
        spatial_distortion_index(preds_t, ms_t, pan_t, pan_lr_t, norm_order, window_size)


@pytest.mark.parametrize(
    ("ms", "pan", "pan_lr", "match"),
    [
        (
            torch.rand((1, 1, 4, 4), dtype=torch.float64),
            torch.rand((1, 1, 16, 16)),
            None,
            "Expected `preds` and `ms` to have the same data type.*",
        ),
        (
            torch.rand((1, 1, 4, 4)),
            torch.rand((1, 1, 16, 16), dtype=torch.float64),
            None,
            "Expected `preds` and `pan` to have the same data type.*",
        ),
        (
            torch.rand((1, 1, 4, 4)),
            torch.rand((1, 1, 16, 16)),
            torch.rand((1, 1, 4, 4), dtype=torch.float64),
            "Expected `preds` and `pan_lr` to have the same data type.*",
        ),
    ],
)
def test_d_s_invalid_type(ms, pan, pan_lr, match):
    """Test that error is raised on different dtypes."""
    preds_t = torch.rand((1, 1, 16, 16))
    with pytest.raises(TypeError, match=match):
        spatial_distortion_index(preds_t, ms, pan, pan_lr, norm_order=1, window_size=7)
