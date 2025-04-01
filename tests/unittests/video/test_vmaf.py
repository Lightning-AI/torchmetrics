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

import pytest
import torch
import vmaf_torch
from einops import rearrange

from torchmetrics.functional.video.vmaf import video_multi_method_assessment_fusion
from torchmetrics.utilities.imports import _TORCH_VMAF_AVAILABLE
from torchmetrics.video import VideoMultiMethodAssessmentFusion
from unittests import _Input
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

seed_all(42)


def _reference_vmaf(preds, target, elementary_features=False):
    """Reference implementation of VMAF metric."""
    device = preds.device
    orig_dtype = preds.dtype

    # Convert to float32 for processing
    preds = (preds.clamp(-1, 1).to(torch.float32) + 1) / 2  # [-1, 1] -> [0, 1]
    target = (target.clamp(-1, 1).to(torch.float32) + 1) / 2  # [-1, 1] -> [0, 1]

    # Calculate luma component
    def calculate_luma(video):
        r = video[:, :, 0, :, :]
        g = video[:, :, 1, :, :]
        b = video[:, :, 2, :, :]
        return (0.299 * r + 0.587 * g + 0.114 * b).unsqueeze(1) * 255  # [0, 1] -> [0, 255]

    preds_luma = calculate_luma(preds)
    target_luma = calculate_luma(target)

    vmaf = vmaf_torch.VMAF().to(device)

    score = vmaf(rearrange(target_luma, "b c t h w -> (b t) c h w"), rearrange(preds_luma, "b c t h w -> (b t) c h w"))

    if elementary_features:
        adm = vmaf.compute_adm_features(
            rearrange(target_luma, "b c t h w -> (b t) c h w"), rearrange(preds_luma, "b c t h w -> (b t) c h w")
        )
        vif = vmaf.compute_vif_features(
            rearrange(target_luma, "b c t h w -> (b t) c h w"), rearrange(preds_luma, "b c t h w -> (b t) c h w")
        )
        motion = vmaf.compute_motion(rearrange(target_luma, "b c t h w -> (b t) c h w"))
        return score.squeeze().to(orig_dtype), adm.to(orig_dtype), vif.to(orig_dtype), motion.squeeze().to(orig_dtype)
    return score.squeeze().to(orig_dtype)


# Define inputs
NUM_BATCHES, BATCH_SIZE = 2, 4
_inputs = []
for size in [32, 64]:
    preds = torch.rand(NUM_BATCHES, BATCH_SIZE, 3, 10, size, size)
    target = torch.rand(NUM_BATCHES, BATCH_SIZE, 3, 10, size, size)
    _inputs.append(_Input(preds=preds, target=target))


@pytest.mark.skipif(not _TORCH_VMAF_AVAILABLE, reason="test requires vmaf-torch")
@pytest.mark.parametrize("preds, target", [(i.preds, i.target) for i in _inputs])
class TestVMAF(MetricTester):
    """Test class for `VideoMultiMethodAssessmentFusion` metric."""

    atol = 1e-3

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_vmaf(self, preds, target, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=VideoMultiMethodAssessmentFusion,
            reference_metric=_reference_vmaf,
        )

    def test_vmaf_functional(self, preds, target):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=video_multi_method_assessment_fusion,
            reference_metric=_reference_vmaf,
        )

    def test_vmaf_elementary_features(self, preds, target):
        """Test that elementary features are returned when requested."""
        # Test functional implementation
        score = video_multi_method_assessment_fusion(preds[0], target[0], elementary_features=True)
        breakpoint()
        assert isinstance(score, tuple)
        assert len(score) == 4  # VMAF score + ADM + VIF + motion
        assert score[0].shape == (BATCH_SIZE,)  # VMAF score shape
        assert score[1].shape == (BATCH_SIZE, 4)  # ADM shape
        assert score[2].shape == (BATCH_SIZE, 4)  # VIF shape
        assert score[3].shape == (BATCH_SIZE,)  # Motion shape

    def test_vmaf_half_cpu(self, preds, target):
        """Test for half precision on CPU."""
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=VideoMultiMethodAssessmentFusion,
            metric_functional=video_multi_method_assessment_fusion,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_vmaf_half_gpu(self, preds, target):
        """Test for half precision on GPU."""
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=VideoMultiMethodAssessmentFusion,
            metric_functional=video_multi_method_assessment_fusion,
        )


@pytest.mark.skipif(_TORCH_VMAF_AVAILABLE, reason="test requires vmaf-torch")
def test_vmaf_raises_error():
    """Test that appropriate error is raised when vmaf-torch is not installed."""
    with pytest.raises(RuntimeError, match="vmaf-torch is not installed"):
        video_multi_method_assessment_fusion(torch.rand(1, 3, 10, 32, 32), torch.rand(1, 3, 10, 32, 32))
