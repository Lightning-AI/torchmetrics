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


_inputs = _Input(
    img1=torch.rand(4, 2, 3, 50, 50),
    img2=torch.rand(4, 2, 3, 50, 50),
)


def _reference_dists(preds: Tensor, target: Tensor, reduction: str) -> Tensor:
    try:
        from DISTS_pytorch import DISTS as reference_dists
    except ImportError:
        pytest.skip("test requires DISTS_pytorch package to be installed")

    ref = reference_dists()
    res = ref(preds, target).detach().cpu().numpy()
    if reduction == "mean":
        return res.mean()
    if reduction == "sum":
        return res.sum()
    return res


class TestDISTS(MetricTester):
    """Test class for `DISTS` metric."""

    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    @pytest.mark.parametrize("ddp", [True, False])
    def test_dists(self, reduction, ddp):
        self.run_class_metric_test(
            ddp=ddp,
            preds=_inputs.img1,
            target=_inputs.img2,
            metric_class=DeepImageStructureAndTextureSimilarity,
            sk_metric=partial(_reference_dists, reduction=reduction),
            dist_sync_on_step=True,
            metric_args={"reduction": reduction},
            check_dist_sync_on_step=True,
            check_batch=True,
        )

    def test_dists_functional(self):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=_inputs.img1,
            target=_inputs.img2,
            metric_functional=deep_image_structure_and_texture_similarity,
            sk_metric=partial(_reference_dists, reduction="mean"),
            metric_args={"reduction": "mean"},
        )

    def test_dists_differentiability(self):
        """Test that the metric is differentiable."""
        self.run_differentiability_test(
            preds=_inputs.img1,
            target=_inputs.img2,
            metric_module=DeepImageStructureAndTextureSimilarity,
            metric_functional=deep_image_structure_and_texture_similarity,
        )

    def test_dists_half_cpu(self):
        """Test for half + cpu support."""
        self.run_precision_test_cpu(
            preds=_inputs.img1, target=_inputs.img2, metric_module=DeepImageStructureAndTextureSimilarity
        )

    def test_dists_half_gpu(self):
        """Test for half + gpu support."""
        self.run_precision_test_gpu(
            preds=_inputs.img1, target=_inputs.img2, metric_module=DeepImageStructureAndTextureSimilarity
        )
