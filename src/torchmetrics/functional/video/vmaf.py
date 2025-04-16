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
            with values in range [0, 1].
        target: Video tensor of shape (batch, channels, frames, height, width). Expected to be in RGB format
            with values in range [0, 1].
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
        >>> preds = torch.rand(2, 3, 10, 32, 32)  # 2 videos, 3 channels, 10 frames, 32x32 resolution
        >>> target = torch.rand(2, 3, 10, 32, 32)  # 2 videos, 3 channels, 10 frames, 32x32 resolution
        >>> video_multi_method_assessment_fusion(preds, target)
        tensor([ 3.9553, 15.2808, 15.0131, 13.7132, 14.0283, 16.9560])
        >>> vmaf_score, adm_score, vif_score, motion_score = video_multi_method_assessment_fusion(
        ...     preds, target, elementary_features=True
        ... )
        >>> vmaf_score
        tensor([ 3.9553, 15.2808, 15.0131, 13.7132, 14.0283, 16.9560])
        >>> adm_score
        tensor([[0.5128, 0.3583, 0.3275, 0.3798],
                [0.5034, 0.3581, 0.3421, 0.4753],
                [0.5161, 0.3987, 0.3176, 0.2830],
                [0.4823, 0.3802, 0.3569, 0.4263],
                [0.4627, 0.4267, 0.2862, 0.5625],
                [0.5576, 0.4112, 0.3703, 0.5333]])
        >>> vif_score
        tensor([[5.5589e-04, 2.3668e-02, 5.9746e-02, 1.8287e-01],
                [6.3305e-04, 2.1592e-02, 3.8126e-02, 1.1630e-01],
                [1.0766e-03, 1.9478e-02, 3.5908e-02, 5.0494e-02],
                [1.4880e-04, 1.6239e-02, 2.6883e-02, 1.5944e-01],
                [2.1966e-04, 2.2355e-02, 6.6175e-02, 5.8169e-02],
                [1.0138e-03, 1.5265e-02, 5.5632e-02, 1.2230e-01]])
        >>> motion_score
        tensor([ 0.0000, 17.7642, 18.1769, 17.5795, 15.6578, 16.0557])

    """
    if not _TORCH_VMAF_AVAILABLE:
        raise RuntimeError("vmaf-torch is not installed. Please install with `pip install torchmetrics[video]`.")

    orig_dtype, device = preds.dtype, preds.device
    preds_luma = calculate_luma(preds)
    target_luma = calculate_luma(target)

    vmaf = vmaf_torch.VMAF().to(device)

    score: Tensor = vmaf(
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
