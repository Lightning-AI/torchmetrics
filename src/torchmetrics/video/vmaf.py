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
from typing import Any, List, Optional, Tuple, Union

import torch
from torch import Tensor

from torchmetrics.functional.video.vmaf import video_multi_method_assessment_fusion
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _TORCH_VMAF_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE


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
      Expected to be in RGB format with values in range [-1, 1] or [0, 1].
    - ``target`` (:class:`~torch.Tensor`): Video tensor of shape ``(batch, channels, frames, height, width)``.
      Expected to be in RGB format with values in range [-1, 1] or [0, 1].

    As output of `forward` and `compute` the metric returns the following output

    If `elementary_features` is False:
        - ``vmaf`` (:class:`~torch.Tensor`): A tensor with the VMAF score for each video in the batch.
          Higher scores indicate better quality, with typical values ranging from 0 to 100.

    If `elementary_features` is True:
        - ``vmaf_score`` (:class:`~torch.Tensor`): The main VMAF score tensor
        - ``adm_score`` (:class:`~torch.Tensor`): The Additive Detail Measure (ADM) score tensor, which measures
          the preservation of details in the video. Shape is (batch * frames, 4) where the 4 values represent
          different detail scales. Higher values indicate better detail preservation.
        - ``vif_score`` (:class:`~torch.Tensor`): The Visual Information Fidelity (VIF) score tensor, which measures
          the preservation of visual information. Shape is (batch * frames, 4) where the 4 values represent different
          frequency bands. Higher values indicate better information preservation.
        - ``motion_score`` (:class:`~torch.Tensor`): The motion score tensor, which measures the amount of motion
          in the video. Shape is (batch * frames,). Higher values indicate more motion.

    Args:
        elementary_features: If True, returns the elementary features used by VMAF.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        RuntimeError: If vmaf-torch is not installed.
        ValueError: If `elementary_features` is not a boolean.

    Example:
        >>> from torch import rand
        >>> from torchmetrics.video import VideoMultiMethodAssessmentFusion
        >>> preds = rand(2, 3, 10, 32, 32)
        >>> target = rand(2, 3, 10, 32, 32)
        >>> vmaf = VideoMultiMethodAssessmentFusion()
        >>> vmaf(preds, target)
        tensor([12.6859, 15.1940, 14.6993, 14.9718, 19.1301, 17.1650])
        >>> vmaf = VideoMultiMethodAssessmentFusion(elementary_features=True)
        >>> vmaf_score, adm_score, vif_score, motion_score = vmaf(preds, target)
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

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 100.0  # Updated to match VMAF score range

    vmaf_score: List[Tensor]
    adm_features: List[Tensor]
    vif_features: List[Tensor]
    motion: List[Tensor]

    def __init__(self, elementary_features: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not _TORCH_VMAF_AVAILABLE:
            raise RuntimeError("vmaf-torch is not installed. Please install with `pip install torchmetrics[video]`.")

        if not isinstance(elementary_features, bool):
            raise ValueError("Argument `elementary_features` should be a boolean, but got {elementary_features}.")
        self.elementary_features = elementary_features

        self.add_state("vmaf_score", default=[], dist_reduce_fx="cat")
        if self.elementary_features:
            self.add_state("adm_features", default=[], dist_reduce_fx="cat")
            self.add_state("vif_features", default=[], dist_reduce_fx="cat")
            self.add_state("motion", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        score = video_multi_method_assessment_fusion(preds, target, self.elementary_features)
        if self.elementary_features:
            self.vmaf_score.append(score[0])
            self.adm_features.append(score[1])
            self.vif_features.append(score[2])
            self.motion.append(score[3])
        else:
            self.vmaf_score.append(score)

    def compute(self) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]:
        """Compute final VMAF score."""
        if self.elementary_features:
            return (
                torch.cat(self.vmaf_score, dim=0),
                torch.cat(self.adm_features, dim=0),
                torch.cat(self.vif_features, dim=0),
                torch.cat(self.motion, dim=0),
            )
        return torch.cat(self.vmaf_score, dim=0)

    def plot(
        self, val: Optional[Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward()` or `metric.compute()`, or the results
                from multiple calls of `metric.forward()` or `metric.compute()`. If no value is provided, will
                automatically call `metric.compute()` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75
            :caption: Example of plotting VMAF scores and elementary features.

            >>> # Example plotting a single value
            >>> from torch import rand
            >>> from torchmetrics.video import VideoMultiMethodAssessmentFusion
            >>> metric = VideoMultiMethodAssessmentFusion(elementary_features=True)
            >>> preds = rand(2, 3, 10, 32, 32)
            >>> target = rand(2, 3, 10, 32, 32)
            >>> metric.update(preds, target)
            >>> fig_, ax_ = metric.plot()

        """
        return super().plot(val=val, ax=ax)
