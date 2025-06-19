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
from typing import Dict, Union

import torch
from torch import Tensor

from torchmetrics.utilities.imports import _EINOPS_AVAILABLE, _TORCH_VMAF_AVAILABLE

if _TORCH_VMAF_AVAILABLE:
    import pandas as pd  # pandas is installed as a dependency of vmaf-torch
    from vmaf_torch import VMAF
else:
    __doctest_skip__ = ["video_multi_method_assessment_fusion"]

if _EINOPS_AVAILABLE:
    from einops import rearrange


def calculate_luma(video: Tensor) -> Tensor:
    """Calculate the luma component of a video tensor."""
    r = video[:, 0, :, :, :]
    g = video[:, 1, :, :, :]
    b = video[:, 2, :, :, :]
    return (0.299 * r + 0.587 * g + 0.114 * b).unsqueeze(1) * 255  # [0, 1] -> [0, 255]


def video_multi_method_assessment_fusion(
    preds: Tensor,
    target: Tensor,
    features: bool = False,
) -> Union[Tensor, Dict[str, Tensor]]:
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
        features: If True, all the elementary features (ADM, VIF, motion) are returned along with the VMAF score in
            a dictionary. This corresponds to the output you would get from the VMAF command line tool with the `--csv`
            option enabled. If False, only the VMAF score is returned as a tensor.

    Returns:
        - If `features` is False, returns a tensor with shape (batch, frame) of VMAF score for each frame in
          each video. Higher scores indicate better quality, with typical values ranging from 0 to 100.

        - If `features` is True, returns a dictionary where each value is a (batch, frame) tensor of the
          corresponding feature. The keys are:
            - 'integer_motion2': Integer motion feature
            - 'integer_motion': Integer motion feature
            - 'integer_adm2': Integer ADM feature
            - 'integer_adm_scale0': Integer ADM feature at scale 0
            - 'integer_adm_scale1': Integer ADM feature at scale 1
            - 'integer_adm_scale2': Integer ADM feature at scale 2
            - 'integer_adm_scale3': Integer ADM feature at scale 3
            - 'integer_vif_scale0': Integer VIF feature at scale 0
            - 'integer_vif_scale1': Integer VIF feature at scale 1
            - 'integer_vif_scale2': Integer VIF feature at scale 2
            - 'integer_vif_scale3': Integer VIF feature at scale 3
            - 'vmaf': VMAF score for each frame in each video

    Example:
        >>> import torch
        >>> from torchmetrics.functional.video import video_multi_method_assessment_fusion
        >>> # 2 videos, 3 channels, 10 frames, 32x32 resolution
        >>> preds = torch.rand(2, 3, 10, 32, 32, generator=torch.manual_seed(42))
        >>> target = torch.rand(2, 3, 10, 32, 32, generator=torch.manual_seed(43))
        >>> video_multi_method_assessment_fusion(preds, target)
        tensor([[ 9.9900, 15.9050, 14.2600, 16.6130, 15.9130, 14.3000, 13.5800, 13.4940, 15.4680, 20.2770],
                [ 6.2500, 11.2990, 17.3000, 11.4560, 19.0600, 14.9340, 14.0540, 14.4130, 12.4680, 14.8180]])
        >>> vmaf_dict = video_multi_method_assessment_fusion(preds, target, features=True)
        >>> # show a couple of features, more features are available
        >>> vmaf_dict['vmaf']
        tensor([[ 9.9900, 15.9050, 14.2600, 16.6130, 15.9130, 14.3000, 13.5800, 13.4940, 15.4680, 20.2770],
                [ 6.2500, 11.2990, 17.3000, 11.4560, 19.0600, 14.9340, 14.0540, 14.4130, 12.4680, 14.8180]])
        >>> vmaf_dict['integer_adm2']
        tensor([[0.4550, 0.4500, 0.3650, 0.4730, 0.4280, 0.3570, 0.3920, 0.4130, 0.3720, 0.4700],
                [0.4170, 0.3900, 0.4370, 0.3700, 0.4540, 0.3850, 0.3790, 0.4790, 0.3870, 0.3940]])
        >>> vmaf_dict['integer_vif_scale0']
        tensor([[0.0010, 0.0010, 0.0010, 0.0000, 0.0000, 0.0020, 0.0020, 0.0010, 0.0000, 0.0010],
                [0.0000, 0.0000, 0.0010, 0.0010, 0.0010, 0.0010, 0.0000, 0.0000, 0.0020, 0.0010]])

    """
    if not _TORCH_VMAF_AVAILABLE:
        raise RuntimeError("vmaf-torch is not installed. Please install with `pip install torchmetrics[video]`.")
    b = preds.shape[0]
    orig_dtype, device = preds.dtype, preds.device
    preds_luma = calculate_luma(preds)
    target_luma = calculate_luma(target)

    vmaf = VMAF().to(device)

    # we need to compute the model for each video separately
    if not features:
        scores = [
            vmaf.compute_vmaf_score(
                rearrange(target_luma[video], "c f h w -> f c h w"), rearrange(preds_luma[video], "c f h w -> f c h w")
            )
            for video in range(b)
        ]
        return torch.cat(scores, dim=1).t().to(orig_dtype)

    scores_and_features = [
        vmaf.table(
            rearrange(target_luma[video], "c f h w -> f c h w"), rearrange(preds_luma[video], "c f h w -> f c h w")
        )
        for video in range(b)
    ]
    dfs = [scores_and_features[video].apply(pd.to_numeric, errors="coerce") for video in range(b)]
    result = [
        {col: torch.tensor(dfs[video][col].values, dtype=orig_dtype) for col in dfs[video].columns if col != "Frame"}
        for video in range(b)
    ]
    return {col: torch.stack([result[video][col] for video in range(b)]) for col in result[0]}
