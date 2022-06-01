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
from collections import namedtuple
from functools import partial

import pytest
import torch
from lpips import LPIPS as LPIPS_reference
from torch import Tensor

from tests.helpers import seed_all
from tests.helpers.testers import BATCH_SIZE, NUM_BATCHES, MetricTester
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.utilities.imports import _LPIPS_AVAILABLE

seed_all(42)

Input = namedtuple("Input", ["img1", "img2"])

_inputs = Input(
    img1=torch.rand(int(NUM_BATCHES * 0.4), int(BATCH_SIZE / 16), 3, 100, 100),
    img2=torch.rand(int(NUM_BATCHES * 0.4), int(BATCH_SIZE / 16), 3, 100, 100),
)


def _compare_fn(img1: Tensor, img2: Tensor, net_type: str, reduction: str = "mean") -> Tensor:
    """comparison function for tm implementation."""
    ref = LPIPS_reference(net=net_type)
    res = ref(img1, img2).detach().cpu().numpy()
    if reduction == "mean":
        return res.mean()
    return res.sum()


@pytest.mark.skipif(not _LPIPS_AVAILABLE, reason="test requires that lpips is installed")
@pytest.mark.parametrize("net_type", ["vgg", "alex", "squeeze"])
class TestLPIPS(MetricTester):
    atol: float = 1e-6

    @pytest.mark.parametrize("ddp", [True, False])
    def test_lpips(self, net_type, ddp):
        """test modular implementation for correctness."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=_inputs.img1,
            target=_inputs.img2,
            metric_class=LearnedPerceptualImagePatchSimilarity,
            sk_metric=partial(_compare_fn, net_type=net_type),
            dist_sync_on_step=False,
            check_scriptable=False,
            check_state_dict=False,
            metric_args={"net_type": net_type},
        )

    def test_lpips_differentiability(self, net_type):
        """test for differentiability of LPIPS metric."""
        self.run_differentiability_test(
            preds=_inputs.img1, target=_inputs.img2, metric_module=LearnedPerceptualImagePatchSimilarity
        )

    # LPIPS half + cpu does not work due to missing support in torch.min
    @pytest.mark.xfail(reason="LPIPS metric does not support cpu + half precision")
    def test_lpips_half_cpu(self, net_type):
        """test for half + cpu support."""
        self.run_precision_test_cpu(_inputs.img1, _inputs.img2, LearnedPerceptualImagePatchSimilarity)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_lpips_half_gpu(self, net_type):
        """test for half + gpu support."""
        self.run_precision_test_gpu(_inputs.img1, _inputs.img2, LearnedPerceptualImagePatchSimilarity)


@pytest.mark.skipif(not _LPIPS_AVAILABLE, reason="test requires that lpips is installed")
def test_error_on_wrong_init():
    """Test class raises the expected errors."""
    with pytest.raises(ValueError, match="Argument `net_type` must be one .*"):
        LearnedPerceptualImagePatchSimilarity(net_type="resnet")

    with pytest.raises(ValueError, match="Argument `reduction` must be one .*"):
        LearnedPerceptualImagePatchSimilarity(reduction=None)


@pytest.mark.skipif(not _LPIPS_AVAILABLE, reason="test requires that lpips is installed")
@pytest.mark.parametrize(
    "inp1, inp2",
    [
        (torch.rand(1, 1, 28, 28), torch.rand(1, 3, 28, 28)),  # wrong number of channels
        (torch.rand(1, 3, 28, 28), torch.rand(1, 1, 28, 28)),  # wrong number of channels
        (torch.randn(1, 3, 28, 28), torch.rand(1, 3, 28, 28)),  # non-normalized input
        (torch.rand(1, 3, 28, 28), torch.randn(1, 3, 28, 28)),  # non-normalized input
    ],
)
def test_error_on_wrong_update(inp1, inp2):
    """test error is raised on wrong input to update method."""
    metric = LearnedPerceptualImagePatchSimilarity()
    with pytest.raises(ValueError, match="Expected both input arguments to be normalized tensors .*"):
        metric(inp1, inp2)
