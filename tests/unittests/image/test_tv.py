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
from typing import Any

import pytest
import torch
from kornia.losses import total_variation as kornia_total_variation
from torchmetrics.functional.image.tv import total_variation
from torchmetrics.image.tv import TotalVariation

from unittests import _Input
from unittests.helpers import seed_all
from unittests.helpers.testers import MetricTester

seed_all(42)


# add extra argument to make the metric and reference fit into the testing framework
class TotalVariationTester(TotalVariation):
    """Tester class for `TotalVariation` metric overriding its update method."""

    def update(self, img, *args: Any):
        """Update metric."""
        super().update(img=img)


def _total_variaion_tester(preds, target, reduction="mean"):
    return total_variation(preds, reduction)


def _total_variation_kornia_tester(preds, target, reduction):
    score = kornia_total_variation(preds).sum(-1)
    if reduction == "sum":
        return score.sum()
    if reduction == "mean":
        return score.mean()
    return score


# define inputs


_inputs = []
for size, channel, dtype in [
    (12, 3, torch.float),
    (13, 3, torch.float32),
    (14, 3, torch.double),
    (15, 3, torch.float64),
]:
    preds = torch.rand(2, 4, channel, size, size, dtype=dtype)
    target = torch.rand(2, 4, channel, size, size, dtype=dtype)
    _inputs.append(_Input(preds=preds, target=target))


@pytest.mark.parametrize(
    "preds, target",
    [(i.preds, i.target) for i in _inputs],
)
@pytest.mark.parametrize("reduction", ["sum", "mean", None])
class TestTotalVariation(MetricTester):
    """Test class for `TotalVariation` metric."""

    @pytest.mark.parametrize("ddp", [True, False])
    def test_total_variation(self, preds, target, reduction, ddp):
        """Test class implementation of metric."""
        if reduction is None and ddp:
            pytest.skip("reduction=None and ddp=True runs out of memory on CI hardware, but it does work")
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            TotalVariationTester,
            partial(_total_variation_kornia_tester, reduction=reduction),
            metric_args={"reduction": reduction},
        )

    def test_total_variation_functional(self, preds, target, reduction):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            _total_variaion_tester,
            partial(_total_variation_kornia_tester, reduction=reduction),
            metric_args={"reduction": reduction},
        )

    def test_sam_half_cpu(self, preds, target, reduction):
        """Test for half precision on CPU."""
        self.run_precision_test_cpu(
            preds,
            target,
            TotalVariationTester,
            _total_variaion_tester,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_sam_half_gpu(self, preds, target, reduction):
        """Test for half precision on GPU."""
        self.run_precision_test_gpu(preds, target, TotalVariationTester, _total_variaion_tester)


def test_correct_args():
    """That that arguments have the right type and sizes."""
    with pytest.raises(ValueError, match="Expected argument `reduction`.*"):
        _ = TotalVariation(reduction="diff")

    with pytest.raises(RuntimeError, match="Expected input `img` to.*"):
        _ = total_variation(torch.randn(1, 2, 3))
