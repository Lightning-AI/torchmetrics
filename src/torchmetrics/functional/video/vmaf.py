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
    r = video[:, 0, :, :, :]
    g = video[:, 1, :, :, :]
    b = video[:, 2, :, :, :]
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
        If `elementary_features` is False, returns a tensor with shape (batch, frame) of VMAF score for each frame in
        each video. Higher scores indicate better quality, with typical values ranging from 0 to 100.

        If `elementary_features` is True, returns a tuple of four tensors:
            - vmaf_score: The main VMAF score tensor of shape (batch, frames)
            - adm_score: The Additive Detail Measure (ADM) score tensor, which measures the preservation of details
              in the video. Shape is (batch, frames, 4) where the 4 values represent different detail scales.
              Higher values indicate better detail preservation.
            - vif_score: The Visual Information Fidelity (VIF) score tensor, which measures the preservation of
              visual information. Shape is (batch, frames, 4) where the 4 values represent different frequency bands.
              Higher values indicate better information preservation.
            - motion_score: The motion score tensor, which measures the amount of motion in the video.
              Shape is (batch, frames). Higher values indicate more motion.

    Example:
        >>> import torch
        >>> from torchmetrics.functional.video import video_multi_method_assessment_fusion
        >>> preds = torch.rand(2, 3, 10, 32, 32)  # 2 videos, 3 channels, 10 frames, 32x32 resolution
        >>> target = torch.rand(2, 3, 10, 32, 32)  # 2 videos, 3 channels, 10 frames, 32x32 resolution
        >>> video_multi_method_assessment_fusion(preds, target)
        tensor([[ 7.0141, 17.4276, 15.1429, 14.9831, 19.3378, 12.5638, 13.9680, 13.4165, 17.9314, 15.6604],
                [13.9790, 11.1951, 15.3990, 13.5877, 15.1370, 18.4508, 17.5596, 18.6859, 12.9309, 15.1975]])
        >>> vmaf_score, adm_score, vif_score, motion_score = video_multi_method_assessment_fusion(
        ...     preds, target, elementary_features=True
        ... )
        >>> vmaf_score
        tensor([[ 7.0141, 17.4276, 15.1429, 14.9831, 19.3378, 12.5638, 13.9680, 13.4165, 17.9314, 15.6604],
                [13.9790, 11.1951, 15.3990, 13.5877, 15.1370, 18.4508, 17.5596, 18.6859, 12.9309, 15.1975]])
        >>> adm_score  # doctest: +NORMALIZE_WHITESPACE
        tensor([[[0.5052, 0.3689, 0.3906, 0.2976],
                 [0.4771, 0.3502, 0.3273, 0.7300],
                 [0.4881, 0.4082, 0.2437, 0.2755],
                 [0.4948, 0.3176, 0.4467, 0.3533],
                 [0.5658, 0.4589, 0.3998, 0.3095],
                 [0.5100, 0.4339, 0.5263, 0.4693],
                 [0.5351, 0.4767, 0.4267, 0.2752],
                 [0.5028, 0.3350, 0.3247, 0.4020],
                 [0.5071, 0.3949, 0.3832, 0.3111],
                 [0.4666, 0.3583, 0.4521, 0.2777]],
               [[0.4686, 0.4700, 0.2433, 0.4896],
                 [0.4952, 0.3658, 0.3985, 0.4379],
                 [0.5445, 0.3839, 0.4010, 0.2285],
                 [0.5038, 0.3151, 0.4543, 0.3893],
                 [0.4899, 0.4008, 0.4266, 0.3279],
                 [0.5109, 0.3921, 0.3264, 0.5778],
                 [0.5315, 0.3788, 0.3103, 0.6088],
                 [0.4607, 0.4334, 0.4077, 0.4407],
                 [0.5017, 0.3816, 0.2890, 0.3553],
                 [0.5284, 0.4586, 0.3681, 0.2760]]])
        >>> vif_score  # doctest: +NORMALIZE_WHITESPACE
        tensor([[[3.9898e-04, 2.7862e-02, 3.1761e-02, 4.5509e-02],
                 [1.6094e-04, 1.1518e-02, 2.0446e-02, 8.4023e-02],
                 [3.7477e-04, 7.8991e-03, 6.6453e-03, 5.7339e-04],
                 [6.7157e-04, 9.9271e-03, 4.6627e-02, 4.6662e-02],
                 [9.6011e-04, 1.3214e-02, 2.7918e-02, 1.6376e-02],
                 [6.7778e-04, 6.1006e-02, 9.8535e-02, 2.5073e-01],
                 [1.1227e-03, 3.3202e-02, 6.4757e-02, 8.6356e-02],
                 [1.2290e-04, 1.3186e-02, 3.0758e-02, 1.0355e-01],
                 [5.8098e-04, 3.3142e-03, 7.3332e-04, 5.8651e-04],
                 [2.5460e-04, 5.2497e-03, 1.7505e-02, 3.1771e-02]],
                [[3.6456e-04, 1.4340e-02, 2.9021e-02, 1.1958e-01],
                 [1.5903e-04, 3.4139e-02, 1.1511e-01, 1.3284e-01],
                 [9.7763e-04, 9.1875e-03, 2.0795e-02, 7.2092e-02],
                 [4.7811e-04, 3.0047e-02, 5.6494e-02, 1.3386e-01],
                 [1.1665e-03, 1.7940e-02, 5.3484e-02, 1.5105e-01],
                 [9.6759e-04, 1.7089e-02, 2.1730e-02, 7.3590e-03],
                 [4.2169e-04, 1.2152e-02, 1.4762e-02, 5.8642e-02],
                 [1.5370e-04, 1.1013e-02, 1.0387e-02, 1.2726e-02],
                 [1.0364e-03, 2.8013e-02, 3.8921e-02, 7.5270e-02],
                 [8.9485e-04, 2.3440e-02, 4.1318e-02, 9.4294e-02]]])
        >>> motion_score
        tensor([[ 0.0000, 15.9685, 15.9246, 15.7889, 17.3888, 19.0524, 13.7110, 16.0245, 16.1028, 15.5713],
                [14.7679, 15.5407, 15.9964, 17.2818, 18.3270, 19.0149, 16.8640, 16.4841, 16.4464, 17.4890]])

    """
    if not _TORCH_VMAF_AVAILABLE:
        raise RuntimeError("vmaf-torch is not installed. Please install with `pip install torchmetrics[video]`.")
    b, f = preds.shape[0], preds.shape[2]
    orig_dtype, device = preds.dtype, preds.device
    preds_luma = calculate_luma(preds)
    target_luma = calculate_luma(target)

    vmaf = vmaf_torch.VMAF().to(device)

    score: Tensor = vmaf(
        rearrange(target_luma, "b c f h w -> (b f) c h w"), rearrange(preds_luma, "b c f h w -> (b f) c h w")
    ).to(orig_dtype)
    score = rearrange(score, "(b f) 1 -> b f", b=b, f=f)
    if not elementary_features:
        return score

    adm = vmaf.compute_adm_features(
        rearrange(target_luma, "b c f h w -> (b f) c h w"), rearrange(preds_luma, "b c f h w -> (b f) c h w")
    )
    adm = rearrange(adm, "(b f) s -> b f s", b=b, f=f)  # s=4 are the different scales
    vif = vmaf.compute_vif_features(
        rearrange(target_luma, "b c f h w -> (b f) c h w"), rearrange(preds_luma, "b c f h w -> (b f) c h w")
    )
    vif = rearrange(vif, "(b f) s -> b f s", b=b, f=f)  # s=4 are the different frequency bands
    motion = vmaf.compute_motion(rearrange(target_luma, "b c f h w -> (b f) c h w"))
    motion = rearrange(motion, "(b f) 1 -> b f", b=b, f=f)
    return score, adm, vif, motion.squeeze()
