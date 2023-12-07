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
    p: int
    ws: int


_inputs = []
for size, channel, p, r, ws, pan_lr_exists, dtype in [
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
            p=p,
            ws=ws,
        )
    )


def _baseline_d_s(
    preds: np.ndarray, ms: np.ndarray, pan: np.ndarray, pan_lr: np.ndarray = None, p: int = 1, ws: int = 7
) -> float:
    """NumPy based implementation of Spatial Distortion Index, which uses UQI of TorchMetrics."""
    pan_degraded = pan_lr
    if pan_degraded is None:
        try:
            pan_degraded = uniform_filter(pan, size=ws, axes=[1, 2])
        except TypeError:
            pan_degraded = np.array(
                [[uniform_filter(pan[i, ..., j], size=ws) for j in range(pan.shape[-1])] for i in range(len(pan))]
            ).transpose((0, 2, 3, 1))
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
    diff = np.abs(m1 - m2) ** p
    return np.mean(diff) ** (1 / p)


def _np_d_s(preds, target, p, ws):
    np_preds = preds.permute(0, 2, 3, 1).cpu().numpy()
    assert isinstance(target, dict), f"Expected `target` to be dict. Got {type(target)}."
    assert "ms" in target, "Expected `target` to contain 'ms'."
    np_ms = target["ms"].permute(0, 2, 3, 1).cpu().numpy()
    assert "pan" in target, "Expected `target` to contain 'pan'."
    np_pan = target["pan"].permute(0, 2, 3, 1).cpu().numpy()
    np_pan_lr = target["pan_lr"].permute(0, 2, 3, 1).cpu().numpy() if "pan_lr" in target else None

    return _baseline_d_s(
        np_preds,
        np_ms,
        np_pan,
        np_pan_lr,
        p=p,
        ws=ws,
    )


@pytest.mark.parametrize(
    "preds, target, p, ws",
    [(i.preds, i.target, i.p, i.ws) for i in _inputs],
)
class TestSpatialDistortionIndex(MetricTester):
    """Test class for `SpatialDistortionIndex` metric."""

    atol = 3e-6

    @pytest.mark.parametrize("ddp", [True, False])
    def test_d_s(self, preds, target, p, ws, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            SpatialDistortionIndex,
            partial(_np_d_s, p=p, ws=ws),
            metric_args={"p": p, "ws": ws},
        )

    def test_d_s_functional(self, preds, target, p, ws):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            spatial_distortion_index,
            partial(_np_d_s, p=p, ws=ws),
            metric_args={"p": p, "ws": ws},
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_d_s_half_gpu(self, preds, target, p, ws):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(preds, target, SpatialDistortionIndex, spatial_distortion_index, {"p": p, "ws": ws})


@pytest.mark.parametrize(
    ("preds", "target", "p", "ws", "match"),
    [
        (
            [1, 16, 16],
            {"ms": [1, 1, 4, 4], "pan": [1, 1, 16, 16]},
            1,
            3,
            "Expected `preds` to have BxCxHxW shape.*",
        ),  # len(preds.shape)
        ([1, 1, 16, 16], {}, 1, 7, r"Expected `target` to have keys \('ms', 'pan'\).*"),  # target.keys()
        (
            [1, 1, 16, 16],
            {"ms": [1, 1, 4, 4]},
            1,
            3,
            r"Expected `target` to have keys \('ms', 'pan'\).*",
        ),  # target.keys()
        (
            [1, 1, 16, 16],
            {"pan": [1, 1, 16, 16]},
            1,
            3,
            r"Expected `target` to have keys \('ms', 'pan'\).*",
        ),  # target.keys()
        (
            [1, 1, 16, 16],
            {"ms": [1, 4, 4], "pan": [1, 1, 16, 16]},
            1,
            3,
            "Expected `ms` to have BxCxHxW shape.*",
        ),  # len(target.shape)
        (
            [1, 1, 16, 16],
            {"ms": [1, 1, 4, 4], "pan": [1, 16, 16]},
            1,
            3,
            "Expected `pan` to have BxCxHxW shape.*",
        ),  # len(target.shape)
        (
            [1, 1, 16, 16],
            {"ms": [1, 1, 4, 4], "pan": [1, 1, 16, 16], "pan_lr": [1, 4, 4]},
            1,
            3,
            "Expected `pan_lr` to have BxCxHxW shape.*",
        ),  # len(target.shape)
        (
            [1, 1, 16, 16],
            {"ms": [1, 1, 4, 4], "pan": [1, 1, 16, 16]},
            0,
            3,
            "Expected `p` to be a positive integer. Got p: 0.",
        ),  # invalid p
        (
            [1, 1, 16, 16],
            {"ms": [1, 1, 4, 4], "pan": [1, 1, 16, 16]},
            -1,
            3,
            "Expected `p` to be a positive integer. Got p: -1.",
        ),  # invalid p
        (
            [1, 1, 16, 16],
            {"ms": [1, 1, 4, 4], "pan": [1, 1, 16, 16]},
            1,
            0,
            "Expected `ws` to be a positive integer. Got ws: 0.",
        ),  # invalid ws
        (
            [1, 1, 16, 16],
            {"ms": [1, 1, 4, 4], "pan": [1, 1, 16, 16]},
            1,
            -1,
            "Expected `ws` to be a positive integer. Got ws: -1.",
        ),  # invalid ws
        (
            [1, 1, 16, 16],
            {"ms": [1, 1, 4, 4], "pan": [1, 1, 17, 16]},
            1,
            3,
            "Expected `preds` and `pan` to have the same height.*",
        ),  # invalid pan_h
        (
            [1, 1, 16, 16],
            {"ms": [1, 1, 4, 4], "pan": [1, 1, 16, 17]},
            1,
            3,
            "Expected `preds` and `pan` to have the same width.*",
        ),  # invalid pan_w
        (
            [1, 1, 16, 16],
            {"ms": [1, 1, 5, 4], "pan": [1, 1, 16, 16]},
            1,
            3,
            "Expected height of `preds` to be multiple of height of `ms`.*",
        ),  # invalid ms_h
        (
            [1, 1, 16, 16],
            {"ms": [1, 1, 4, 5], "pan": [1, 1, 16, 16]},
            1,
            3,
            "Expected width of `preds` to be multiple of width of `ms`.*",
        ),  # invalid ms_w
        (
            [1, 1, 16, 16],
            {"ms": [1, 1, 4, 4], "pan": [1, 1, 16, 16], "pan_lr": [1, 1, 5, 4]},
            1,
            3,
            "Expected `ms` and `pan_lr` to have the same height.*",
        ),  # invalid pan_lr_h
        (
            [1, 1, 16, 16],
            {"ms": [1, 1, 4, 4], "pan": [1, 1, 16, 16], "pan_lr": [1, 1, 4, 5]},
            1,
            3,
            "Expected `ms` and `pan_lr` to have the same width.*",
        ),  # invalid pan_lr_w
        (
            [1, 1, 16, 16],
            {"ms": [1, 2, 4, 4], "pan": [1, 1, 16, 16]},
            1,
            3,
            "Expected `preds` and `ms` to have same batch and channel.*",
        ),  # invalid ms.shape
        (
            [1, 1, 16, 16],
            {"ms": [2, 1, 4, 4], "pan": [1, 1, 16, 16]},
            1,
            3,
            "Expected `preds` and `ms` to have same batch and channel.*",
        ),  # invalid ms.shape
        (
            [1, 1, 16, 16],
            {"ms": [1, 1, 4, 4], "pan": [1, 2, 16, 16]},
            1,
            3,
            "Expected `preds` and `pan` to have same batch and channel.*",
        ),  # invalid pan.shape
        (
            [1, 1, 16, 16],
            {"ms": [1, 1, 4, 4], "pan": [2, 1, 16, 16]},
            1,
            3,
            "Expected `preds` and `pan` to have same batch and channel.*",
        ),  # invalid pan.shape
        (
            [1, 1, 16, 16],
            {"ms": [1, 1, 4, 4], "pan": [1, 1, 16, 16], "pan_lr": [1, 2, 4, 4]},
            1,
            3,
            "Expected `preds` and `pan_lr` to have same batch and channel.*",
        ),  # invalid pan_lr.shape
        (
            [1, 1, 16, 16],
            {"ms": [1, 1, 4, 4], "pan": [1, 1, 16, 16], "pan_lr": [2, 1, 4, 4]},
            1,
            3,
            "Expected `preds` and `pan_lr` to have same batch and channel.*",
        ),  # invalid pan_lr.shape
        (
            [1, 1, 16, 16],
            {"ms": [1, 1, 4, 4], "pan": [1, 1, 16, 16]},
            1,
            5,
            "Expected `ws` to be smaller than dimension of `ms`.*",
        ),  # invalid ws
    ],
)
def test_d_s_invalid_inputs(preds, target, p, ws, match):
    """Test that invalid input raises the correct errors."""
    preds_t = torch.rand(preds)
    target_t = {name: torch.rand(t) for name, t in target.items()}
    with pytest.raises(ValueError, match=match):
        spatial_distortion_index(preds_t, target_t, p, ws)


@pytest.mark.parametrize(
    ("target", "match"),
    [
        (
            {
                "ms": torch.rand((1, 1, 4, 4), dtype=torch.float64),
                "pan": torch.rand((1, 1, 16, 16)),
            },
            "Expected `preds` and `ms` to have the same data type.*",
        ),
        (
            {
                "ms": torch.rand((1, 1, 4, 4)),
                "pan": torch.rand((1, 1, 16, 16), dtype=torch.float64),
            },
            "Expected `preds` and `pan` to have the same data type.*",
        ),
        (
            {
                "ms": torch.rand((1, 1, 4, 4)),
                "pan": torch.rand((1, 1, 16, 16)),
                "pan_lr": torch.rand((1, 1, 4, 4), dtype=torch.float64),
            },
            "Expected `preds` and `pan_lr` to have the same data type.*",
        ),
    ],
)
def test_d_s_invalid_type(target, match):
    """Test that error is raised on different dtypes."""
    preds_t = torch.rand((1, 1, 16, 16))
    with pytest.raises(TypeError, match=match):
        spatial_distortion_index(preds_t, target, p=1, ws=7)