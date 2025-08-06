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
from typing import Any, Dict, List, Union

from torch import Tensor

from torchmetrics.functional.video.vmaf import video_multi_method_assessment_fusion
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _TORCH_VMAF_AVAILABLE

if not _TORCH_VMAF_AVAILABLE:
    __doctest_skip__ = ["VideoMultiMethodAssessmentFusion"]


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
       Install either by cloning the repository and running ``pip install .``
       or with ``pip install torchmetrics[video]``.

    As input to ``forward`` and ``update`` the metric accepts the following input:

        - ``preds`` (:class:`~torch.Tensor`): Video tensor of shape ``(batch, channels, frames, height, width)``.
          Expected to be in RGB format with values in range [0, 1].
        - ``target`` (:class:`~torch.Tensor`): Video tensor of shape ``(batch, channels, frames, height, width)``.
          Expected to be in RGB format with values in range [0, 1].

    As output of ``forward`` and ``compute`` the metric returns the following output ``vmaf`` (:class:`~torch.Tensor`):

        - If ``features`` is False, returns a tensor with shape (batch, frame)
          of VMAF score for each frame in each video. Higher scores indicate better quality, with typical values
          ranging from 0 to 100.
        - If ``features`` is True, returns a dictionary where each value is a (batch, frame) tensor of the
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

    Args:
        features: If True, all the elementary features (ADM, VIF, motion) are returned along with the VMAF score in
            a dictionary. This corresponds to the output you would get from the VMAF command line tool with
            the ``--csv`` option enabled. If False, only the VMAF score is returned as a tensor.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        RuntimeError:
            If vmaf-torch is not installed.
        ValueError:
            If ``features`` is not a boolean.

    Example:
        >>> import torch
        >>> from torchmetrics.video import VideoMultiMethodAssessmentFusion
        >>> # 2 videos, 3 channels, 10 frames, 32x32 resolution
        >>> preds = torch.rand(2, 3, 10, 32, 32, generator=torch.manual_seed(42))
        >>> target = torch.rand(2, 3, 10, 32, 32, generator=torch.manual_seed(43))
        >>> vmaf = VideoMultiMethodAssessmentFusion()
        >>> torch.round(vmaf(preds, target), decimals=2)
        tensor([[ 9.9900, 15.9000, 14.2600, 16.6100, 15.9100, 14.3000, 13.5800, 13.4900, 15.4700, 20.2800],
                [ 6.2500, 11.3000, 17.3000, 11.4600, 19.0600, 14.9300, 14.0500, 14.4100, 12.4700, 14.8200]])
        >>> vmaf = VideoMultiMethodAssessmentFusion(features=True)
        >>> vmaf_dict = vmaf(preds, target)
        >>> vmaf_dict['vmaf'].round(decimals=2)
        tensor([[ 9.9900, 15.9000, 14.2600, 16.6100, 15.9100, 14.3000, 13.5800, 13.4900, 15.4700, 20.2800],
                [ 6.2500, 11.3000, 17.3000, 11.4600, 19.0600, 14.9300, 14.0500, 14.4100, 12.4700, 14.8200]])
        >>> vmaf_dict['integer_adm2'].round(decimals=2)
        tensor([[0.4500, 0.4500, 0.3600, 0.4700, 0.4300, 0.3600, 0.3900, 0.4100, 0.3700, 0.4700],
                [0.4200, 0.3900, 0.4400, 0.3700, 0.4500, 0.3900, 0.3800, 0.4800, 0.3900, 0.3900]])

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
        if self.features and isinstance(score, dict):
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
        elif isinstance(score, Tensor):
            self.vmaf_score.append(score)

    def compute(self) -> Union[Tensor, Dict[str, Tensor]]:
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
