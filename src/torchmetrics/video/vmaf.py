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
from typing import Any, List

import torch
import vmaf_torch
from torch import Tensor

from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _TORCH_VMAF_AVAILABLE


class VideoMultiMethodAssessmentFusion(Metric):
    """Calculates Video Multi-Method Assessment Fusion (VMAF) metric.

    VMAF combined multiple quality assessment features such as detail loss, motion, and contrast using a machine
    learning model to predict human perception of video quality more accurately than traditional metrics like PSNR
    or SSIM.

    .. note::
        This implementation requires you to have vmaf-torch installed: https://github.com/alvitrioliks/VMAF-torch.
        Install either by cloning the repository and running `pip install .` or with `pip install torchmetrics[video]`.

    Raises:
        ValueError: If vmaf-torch is not installed.

    """

    vmaf_score: List[Tensor]
    adm_features: List[Tensor]
    vif_features: List[Tensor]
    motion: List[Tensor]

    def __init__(self, elementary_features: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not _TORCH_VMAF_AVAILABLE:
            raise RuntimeError("vmaf-torch is not installed. Please install with `pip install torchmetrics[video]`.")

        self.backend = vmaf_torch.VMAF().to(self.device)
        self.backend.compile()
        if not isinstance(elementary_features, bool):
            raise ValueError("Argument `elementary_features` should be a boolean, but got {elementary_features}.")
        self.elementary_features = elementary_features

        self.add_state("vmaf_score", default=[], dist_reduce_fx=None)
        if self.elementary_features:
            self.add_state("adm_features", default=[], dist_reduce_fx=None)
            self.add_state("vif_features", default=[], dist_reduce_fx=None)
            self.add_state("motion", default=[], dist_reduce_fx=None)

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Calculate VMAF score for each video in the batch."""
        self.vmaf_score.append(self.backend(ref=target, dist=preds))
        if self.elementary_features:
            self.adm_features.append(self.backend.compute_adm_features(ref=target, dist=preds))
            self.vif_features.append(self.backend.compute_vif_features(ref=target, dist=preds))
            self.motion.append(self.backend.compute_motion(ref=target))

    def compute(self) -> Tensor:
        """Return the VMAF score for each video in the batch."""
        if self.elementary_features:
            return torch.cat([self.vmaf_score, self.adm_features, self.vif_features, self.motion], dim=1)
        return self.vmaf_score
