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
from typing import Any, List, Tuple, Union

from torch import Tensor

from torchmetrics.functional.video.vmaf import video_multi_method_assessment_fusion
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _TORCH_VMAF_AVAILABLE


class VideoMultiMethodAssessmentFusion(Metric):
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

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``preds`` (:class:`~torch.Tensor`): Video tensor of shape ``(batch, channels, frames, height, width)``.
      Expected to be in RGB format with values in range [0, 1].
    - ``target`` (:class:`~torch.Tensor`): Video tensor of shape ``(batch, channels, frames, height, width)``.
      Expected to be in RGB format with values in range [0, 1].

    As output of `forward` and `compute` the metric returns the following output

    - ``vmaf`` (:class:`~torch.Tensor`): If `elementary_features` is False, returns a tensor with shape (batch, frame)
        of VMAF score for each frame in each video. Higher scores indicate better quality, with typical values ranging
        from 0 to 100.

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

    Args:
        elementary_features: If True, returns the elementary features used by VMAF.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        RuntimeError:
            If vmaf-torch is not installed.
        ValueError:
            If `elementary_features` is not a boolean.

    Example:
        >>> from torch import rand
        >>> from torchmetrics.video import VideoMultiMethodAssessmentFusion
        >>> preds = rand(2, 3, 10, 32, 32)  # 2 videos, 3 channels, 10 frames, 32x32 resolution
        >>> target = rand(2, 3, 10, 32, 32)  # 2 videos, 3 channels, 10 frames, 32x32 resolution
        >>> vmaf = VideoMultiMethodAssessmentFusion()
        >>> vmaf(preds, target)
        tensor([[ 7.0141, 17.4276, 15.1429, 14.9831, 19.3378, 12.5638, 13.9680, 13.4165, 17.9314, 15.6604],
                [13.9790, 11.1951, 15.3990, 13.5877, 15.1370, 18.4508, 17.5596, 18.6859, 12.9309, 15.1975]])
        >>> vmaf = VideoMultiMethodAssessmentFusion(elementary_features=True)
        >>> vmaf_score, adm_score, vif_score, motion_score = vmaf(preds, target)
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

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 100.0

    vmaf_score: List[Tensor]
    integer_motion2: List[Tensor]
    integer_motion: List[Tensor]
    integer_adm2: List[Tensor]
    integer_adm_scale0: List[Tensor]
    integer_adm_scale1: List[Tensor]
    integer_adm_scale2: List[Tensor]
    integer_adm_scale3: List[Tensor]
    integer_vif_scale0: List[Tensor]
    integer_vif_scale1: List[Tensor]
    integer_vif_scale2: List[Tensor]
    integer_vif_scale3: List[Tensor]

    def __init__(self, features: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not _TORCH_VMAF_AVAILABLE:
            raise RuntimeError("vmaf-torch is not installed. Please install with `pip install torchmetrics[video]`.")

        if not isinstance(features, bool):
            raise ValueError("Argument `elementary_features` should be a boolean, but got {features}.")
        self.features = features

        self.add_state("vmaf_score", default=[], dist_reduce_fx="cat")
        if self.features:
            self.add_state("integer_motion2", default=[], dist_reduce_fx="cat")
            self.add_state("integer_motion", default=[], dist_reduce_fx="cat")
            self.add_state("integer_adm2", default=[], dist_reduce_fx="cat")
            self.add_state("integer_adm_scale0", default=[], dist_reduce_fx="cat")
            self.add_state("integer_adm_scale1", default=[], dist_reduce_fx="cat")
            self.add_state("integer_adm_scale2", default=[], dist_reduce_fx="cat")
            self.add_state("integer_adm_scale3", default=[], dist_reduce_fx="cat")
            self.add_state("integer_vif_scale0", default=[], dist_reduce_fx="cat")
            self.add_state("integer_vif_scale1", default=[], dist_reduce_fx="cat")
            self.add_state("integer_vif_scale2", default=[], dist_reduce_fx="cat")
            self.add_state("integer_vif_scale3", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        score = video_multi_method_assessment_fusion(preds, target, self.features)
        if self.features:
            self.vmaf_score.append(score["vmaf"])
            self.integer_motion2.append(score["integer_motion2"])
            self.integer_motion.append(score["integer_motion"])
            self.integer_adm2.append(score["integer_adm2"])
            self.integer_adm_scale0.append(score["integer_adm_scale0"])
            self.integer_adm_scale1.append(score["integer_adm_scale1"])
            self.integer_adm_scale2.append(score["integer_adm_scale2"])
            self.integer_adm_scale3.append(score["integer_adm_scale3"])
            self.integer_vif_scale0.append(score["integer_vif_scale0"])
            self.integer_vif_scale1.append(score["integer_vif_scale1"])
            self.integer_vif_scale2.append(score["integer_vif_scale2"])
            self.integer_vif_scale3.append(score["integer_vif_scale3"])
        else:
            self.vmaf_score.append(score)

    def compute(self) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]:
        """Compute final VMAF score."""
        if self.features:
            return {
                "vmaf": dim_zero_cat(self.vmaf_score),
                "integer_motion2": dim_zero_cat(self.integer_motion2),
                "integer_motion": dim_zero_cat(self.integer_motion),
                "integer_adm2": dim_zero_cat(self.integer_adm2),
                "integer_adm_scale0": dim_zero_cat(self.integer_adm_scale0),
                "integer_adm_scale1": dim_zero_cat(self.integer_adm_scale1),
                "integer_adm_scale2": dim_zero_cat(self.integer_adm_scale2),
                "integer_adm_scale3": dim_zero_cat(self.integer_adm_scale3),
                "integer_vif_scale0": dim_zero_cat(self.integer_vif_scale0),
                "integer_vif_scale1": dim_zero_cat(self.integer_vif_scale1),
                "integer_vif_scale2": dim_zero_cat(self.integer_vif_scale2),
                "integer_vif_scale3": dim_zero_cat(self.integer_vif_scale3),
            }
        return dim_zero_cat(self.vmaf_score)
