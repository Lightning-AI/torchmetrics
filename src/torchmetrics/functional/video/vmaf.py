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
import torch
import vmaf_torch
from einops import rearrange
from torch import Tensor


def calculate_luma(video: Tensor) -> Tensor:
    """Calculate the luma component of a video tensor."""
    r = video[:, :, 0, :, :]
    g = video[:, :, 1, :, :]
    b = video[:, :, 2, :, :]
    return (0.299 * r + 0.587 * g + 0.114 * b).unsqueeze(1) * 255  # [0, 1] -> [0, 255]


def video_multi_method_assessment_fusion(
    preds: Tensor,
    target: Tensor,
    elementary_features: bool = False,
) -> Tensor:
    """Calculates Video Multi-Method Assessment Fusion (VMAF) metric.

    VMAF combined multiple quality assessment features such as detail loss, motion, and contrast using a machine
    learning model to predict human perception of video quality more accurately than traditional metrics like PSNR
    or SSIM.

    .. note::
        This implementation requires you to have vmaf-torch installed: https://github.com/alvitrioliks/VMAF-torch.
        Install either by cloning the repository and running `pip install .` or with `pip install torchmetrics[video]`.

    Args:
        preds: Video tensor of shape (batch, channels, frames, height, width).
        target: Video tensor of shape (batch, channels, frames, height, width).
        elementary_features: If True, returns the elementary features used by VMAF.

    Returns:
        If `elementary_features` is False, returns a tensor with the VMAF score for each video in the batch.
        If `elementary_features` is True, returns a tensor with the VMAF score and the elementary features used by VMAF.

    Example:
        >>> import torch
        >>> from torchmetrics.functional.video import video_multi_method_assessment_fusion
        >>> preds = torch.rand(2, 3, 10, 32, 32)
        >>> target = torch.rand(2, 3, 10, 32, 32)
        >>> vmaf = video_multi_method_assessment_fusion(preds, target)
        torch.tensor([0.0, 0.0])

    """
    orig_dtype = preds.dtype
    device = preds.device

    preds = (preds.clamp(-1, 1).to(torch.float32) + 1) / 2  # [-1, 1] -> [0, 1]
    target = (target.clamp(-1, 1).to(torch.float32) + 1) / 2  # [-1, 1] -> [0, 1]

    preds_luma = calculate_luma(preds)
    target_luma = calculate_luma(target)

    vmaf = vmaf_torch.VMAF().to(device)

    score = vmaf(
        rearrange(target_luma, "b c t h w -> (b t) c h w"), rearrange(preds_luma, "b c t h w -> (b t) c h w")
    ).to(orig_dtype)

    if elementary_features:
        adm = vmaf.compute_adm_features(
            rearrange(target_luma, "b c t h w -> (b t) c h w"), rearrange(preds_luma, "b c t h w -> (b t) c h w")
        )
        vif = vmaf.compute_vif_features(
            rearrange(target_luma, "b c t h w -> (b t) c h w"), rearrange(preds_luma, "b c t h w -> (b t) c h w")
        )
        motion = vmaf.compute_motion(rearrange(target_luma, "b c t h w -> (b t) c h w"))
        score = torch.cat([score, adm, vif, motion], dim=-1)
    return score
