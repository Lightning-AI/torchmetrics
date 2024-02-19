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
from torch import Tensor
from torchmetrics.functional.image.qnr import quality_with_no_reference
from torchmetrics.image.qnr import QualityWithNoReference

from unittests import BATCH_SIZE, NUM_BATCHES
from unittests.helpers import seed_all
from unittests.helpers.testers import MetricTester
from unittests.image.test_d_lambda import _baseline_d_lambda
from unittests.image.test_d_s import _baseline_d_s

seed_all(42)


class _Input(NamedTuple):
    preds: Tensor
    target: List[Dict[str, Tensor]]
    ms: Tensor
    pan: Tensor
    pan_lr: Tensor
    alpha: float
    beta: float
    norm_order: int
    window_size: int


_inputs = []
for size, channel, alpha, beta, norm_order, r, window_size, pan_lr_exists, dtype in [
    (12, 3, 1, 1, 1, 16, 3, False, torch.float),
    (13, 1, 1, 1, 3, 8, 5, False, torch.float32),
    (14, 1, 1, 1, 4, 4, 5, True, torch.double),
    (15, 3, 1, 1, 1, 2, 3, True, torch.float64),
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
            alpha=alpha,
            beta=beta,
            norm_order=norm_order,
            window_size=window_size,
        )
    )


def _baseline_quality_with_no_reference(
    preds: np.ndarray,
    ms: np.ndarray,
    pan: np.ndarray,
    pan_lr: np.ndarray = None,
    alpha: float = 1,
    beta: float = 1,
    norm_order: int = 1,
    window_size: int = 7,
) -> float:
    """NumPy based implementation of Quality with No Reference, which uses D_lambda and D_s."""
    d_lambda = _baseline_d_lambda(preds, ms, norm_order)
    d_s = _baseline_d_s(preds, ms, pan, pan_lr, norm_order, window_size)
    return (1 - d_lambda) ** alpha * (1 - d_s) ** beta


def _np_quality_with_no_reference(preds, target, pan=None, pan_lr=None, alpha=1, beta=1, norm_order=1, window_size=7):
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

    return _baseline_quality_with_no_reference(
        np_preds,
        np_ms,
        np_pan,
        np_pan_lr,
        alpha=alpha,
        beta=beta,
        norm_order=norm_order,
        window_size=window_size,
    )


def _invoke_quality_with_no_reference(preds, target, ms, pan, pan_lr, alpha, beta, norm_order, window_size):
    ms = target.get("ms", ms)
    pan = target.get("pan", pan)
    pan_lr = target.get("pan_lr", pan_lr)
    return quality_with_no_reference(preds, ms, pan, pan_lr, alpha, beta, norm_order, window_size)


@pytest.mark.parametrize(
    "preds, target, ms, pan, pan_lr, alpha, beta, norm_order, window_size",
    [(i.preds, i.target, i.ms, i.pan, i.pan_lr, i.alpha, i.beta, i.norm_order, i.window_size) for i in _inputs],
)
class TestQualityWithNoReference(MetricTester):
    """Test class for `QualityWithNoReference` metric."""

    atol = 3e-6

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_quality_with_no_reference(self, preds, target, ms, pan, pan_lr, alpha, beta, norm_order, window_size, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            QualityWithNoReference,
            partial(_np_quality_with_no_reference, norm_order=norm_order, window_size=window_size),
            metric_args={"alpha": alpha, "beta": beta, "norm_order": norm_order, "window_size": window_size},
        )

    def test_quality_with_no_reference_functional(
        self, preds, target, ms, pan, pan_lr, alpha, beta, norm_order, window_size
    ):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            ms,
            quality_with_no_reference,
            partial(
                _np_quality_with_no_reference, alpha=alpha, beta=beta, norm_order=norm_order, window_size=window_size
            ),
            metric_args={"alpha": alpha, "beta": beta, "norm_order": norm_order, "window_size": window_size},
            fragment_kwargs=True,
            pan=pan,
            pan_lr=pan_lr,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_quality_with_no_reference_half_gpu(
        self, preds, target, ms, pan, pan_lr, alpha, beta, norm_order, window_size
    ):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(
            preds,
            target,
            QualityWithNoReference,
            partial(
                _invoke_quality_with_no_reference,
                ms=ms,
                pan=pan,
                pan_lr=pan_lr,
                alpha=alpha,
                beta=beta,
                norm_order=norm_order,
                window_size=window_size,
            ),
            {"alpha": alpha, "beta": beta, "norm_order": norm_order, "window_size": window_size},
        )


@pytest.mark.parametrize(
    ("alpha", "beta", "match"),
    [
        (-1, 1, "Expected `alpha` to be a non-negative real number. Got alpha: -1."),  # invalid alpha
        (1, -1, "Expected `beta` to be a non-negative real number. Got beta: -1."),  # invalid beta
    ],
)
def test_quality_with_no_reference_invalid_inputs(alpha, beta, match):
    """Test that invalid input raises the correct errors."""
    preds = torch.rand((1, 1, 16, 16))
    ms = torch.rand((1, 1, 4, 4))
    pan = torch.rand((1, 1, 16, 16))
    pan_lr = None
    norm_order = 1
    window_size = 3
    with pytest.raises(ValueError, match=match):
        quality_with_no_reference(preds, ms, pan, pan_lr, alpha, beta, norm_order, window_size)


def test_quality_with_no_reference_with_zero_alpha_and_beta():
    """Test QNR to be exactly 1 when both alpha and beta are 0."""
    preds = torch.rand((1, 1, 32, 32))
    ms = torch.rand((1, 1, 16, 16))
    pan = torch.rand((1, 1, 32, 32))
    pan_lr = None
    alpha = 0
    beta = 0
    norm_order = 1
    window_size = 3
    assert quality_with_no_reference(preds, ms, pan, pan_lr, alpha, beta, norm_order, window_size) == 1


@pytest.mark.parametrize(
    ("alpha", "beta"),
    [
        (0, 1e8),
        (1e8, 0),
        (1e8, 1e8),
    ],
)
def test_quality_with_no_reference_within_range(alpha, beta):
    """Test QNR to be in the range of 0 and 1."""
    preds = torch.rand((1, 1, 32, 32))
    ms = torch.rand((1, 1, 16, 16))
    pan = torch.rand((1, 1, 32, 32))
    pan_lr = None
    norm_order = 1
    window_size = 3
    qnr = quality_with_no_reference(preds, ms, pan, pan_lr, alpha, beta, norm_order, window_size)
    assert qnr >= 0
    assert qnr <= 1
