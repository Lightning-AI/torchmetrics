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
from typing import Tuple, Union

import torch
import vmaf_torch
from einops import rearrange
from torch import Tensor

from torchmetrics.utilities.imports import _TORCH_VMAF_AVAILABLE


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
) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]:
    """Calculates Video Multi-Method Assessment Fusion (VMAF) metric.

    VMAF is a full-reference video quality assessment algorithm that combines multiple quality assessment features
    such as detail loss, motion, and contrast using a machine learning model to predict human perception of video
    quality more accurately than traditional metrics like PSNR or SSIM.

    The metric works by:
    1. Converting input videos to luma component (grayscale)
    2. Computing multiple elementary features:
       - Additive Detail Measure (ADM): Evaluates detail preservation at different scales
       - Visual Information Fidelity (VIF): Measures preservation of visual information across frequency bands
       - Motion: Quantifies the amount of motion in the video
    3. Combining these features using a trained SVM model to predict quality

    .. note::
        This implementation requires you to have vmaf-torch installed: https://github.com/alvitrioliks/VMAF-torch.
        Install either by cloning the repository and running `pip install .` or with `pip install torchmetrics[video]`.

    Args:
        preds: Video tensor of shape (batch, channels, frames, height, width). Expected to be in RGB format
            with values in range [-1, 1] or [0, 1].
        target: Video tensor of shape (batch, channels, frames, height, width). Expected to be in RGB format
            with values in range [-1, 1] or [0, 1].
        elementary_features: If True, returns the elementary features used by VMAF.

    Returns:
        If `elementary_features` is False, returns a tensor with the VMAF score for each video in the batch.
        Higher scores indicate better quality, with typical values ranging from 0 to 100.

        If `elementary_features` is True, returns a tuple of four tensors:
            - vmaf_score: The main VMAF score tensor
            - adm_score: The Additive Detail Measure (ADM) score tensor, which measures the preservation of details
              in the video. Shape is (batch * frames, 4) where the 4 values represent different detail scales.
              Higher values indicate better detail preservation.
            - vif_score: The Visual Information Fidelity (VIF) score tensor, which measures the preservation of
              visual information. Shape is (batch * frames, 4) where the 4 values represent different frequency bands.
              Higher values indicate better information preservation.
            - motion_score: The motion score tensor, which measures the amount of motion in the video.
              Shape is (batch * frames,). Higher values indicate more motion.

    Example:
        >>> import torch
        >>> from torchmetrics.functional.video import video_multi_method_assessment_fusion
        >>> preds = torch.rand(2, 3, 10, 32, 32)
        >>> target = torch.rand(2, 3, 10, 32, 32)
        >>> video_multi_method_assessment_fusion(preds, target)
        tensor([12.6859, 15.1940, 14.6993, 14.9718, 19.1301, 17.1650])
        >>> vmaf_score, adm_score, vif_score, motion_score = video_multi_method_assessment_fusion(
        ...     preds, target, elementary_features=True
        ... )
        >>> vmaf_score
        tensor([12.6859, 15.1940, 14.6993, 14.9718, 19.1301, 17.1650])
        >>> adm_score
        tensor([[0.6258, 0.4526, 0.4360, 0.5100],
                [0.6117, 0.4558, 0.4478, 0.5543],
                [0.6253, 0.4867, 0.4116, 0.4412],
                [0.6011, 0.4773, 0.4527, 0.5263],
                [0.5830, 0.5209, 0.4050, 0.6781],
                [0.6576, 0.5081, 0.4600, 0.6017]])
        >>> vif_score
        tensor([[6.8940e-04, 3.5287e-02, 1.2094e-01, 6.7600e-01],
                [7.8453e-04, 3.1258e-02, 6.3257e-02, 3.4321e-01],
                [1.3337e-03, 2.8432e-02, 6.3114e-02, 4.6726e-01],
                [1.8480e-04, 2.3861e-02, 1.5634e-01, 5.5803e-01],
                [2.7257e-04, 3.4004e-02, 1.6240e-01, 6.9619e-01],
                [1.2596e-03, 2.1799e-02, 1.0870e-01, 2.2582e-01]])
        >>> motion_score
        tensor([0.0000, 8.8821, 9.0885, 8.7898, 7.8289, 8.0279])

    """
    if not _TORCH_VMAF_AVAILABLE:
        raise RuntimeError("vmaf-torch is not installed. Please install with `pip install torchmetrics[video]`.")

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
        return score.squeeze(), adm, vif, motion.squeeze()
    return score.squeeze()
