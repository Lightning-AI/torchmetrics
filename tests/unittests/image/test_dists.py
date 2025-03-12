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
from torch import Tensor

from torchmetrics.functional.image.dists import deep_image_structure_and_texture_similarity
from torchmetrics.image.dists import DeepImageStructureAndTextureSimilarity
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

seed_all(42)


class _Input(NamedTuple):
    img1: Tensor
    img2: Tensor


_input1 = _Input(
    img1=torch.rand(4, 2, 3, 50, 50),
    img2=torch.rand(4, 2, 3, 50, 50),
)

_input2 = _Input(
    img1=torch.rand(4, 2, 3, 100, 100),
    img2=torch.rand(4, 2, 3, 100, 100),
)


def _reference_dists(preds: Tensor, target: Tensor, reduction: str) -> Tensor:
    """Compute DISTS using the reference implementation."""
    try:
        from DISTS_pytorch import DISTS as reference_dists  # noqa: N811
    except ImportError:
        pytest.skip("test requires DISTS_pytorch package to be installed")

    ref = reference_dists()
    res = ref(preds, target).detach().cpu().numpy()
    if reduction == "mean":
        return res.mean()
    if reduction == "sum":
        return res.sum()
    return res


@pytest.mark.parametrize("inputs", [_input1, _input2])
class TestDISTS(MetricTester):
    """Test class for `DISTS` metric."""

    @pytest.mark.parametrize("ddp", [True, False])
    def test_dists(self, inputs: _Input, ddp: bool) -> None:
        """Test modular implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=inputs.img1,
            target=inputs.img2,
            metric_class=DeepImageStructureAndTextureSimilarity,
            reference_metric=partial(_reference_dists, reduction="mean"),
        )

    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    def test_dists_functional(self, inputs: _Input, reduction: str) -> None:
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=inputs.img1,
            target=inputs.img2,
            metric_functional=deep_image_structure_and_texture_similarity,
            reference_metric=partial(_reference_dists, reduction=reduction),
            metric_args={"reduction": reduction},
        )

    def test_dists_differentiability(self, inputs: _Input):
        """Test that the metric is differentiable."""
        self.run_differentiability_test(
            preds=inputs.img1,
            target=inputs.img2,
            metric_module=DeepImageStructureAndTextureSimilarity,
            metric_functional=deep_image_structure_and_texture_similarity,
        )

    def test_dists_half_cpu(self, inputs: _Input):
        """Test for half + cpu support."""
        self.run_precision_test_cpu(
            preds=inputs.img1, target=inputs.img2, metric_module=DeepImageStructureAndTextureSimilarity
        )

    def test_dists_half_gpu(self, inputs: _Input):
        """Test for half + gpu support."""
        self.run_precision_test_gpu(
            preds=inputs.img1, target=inputs.img2, metric_module=DeepImageStructureAndTextureSimilarity
        )
