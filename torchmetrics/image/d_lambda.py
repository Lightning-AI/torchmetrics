# Copyright The PyTorch Lightning team.
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
from typing import Any, List, Optional

from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.image.d_lambda import _d_lambda_compute, _d_lambda_update
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat


class SpectralDistortionIndex(Metric):
    """Computes Spectral Distortion Index (SpectralDistortionIndex_).

    Args:
        ms: Low resolution multispectral image
        fused: High resolution fused image
        p: Large spectral differences (default: 1)
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied


    Return:
        Tensor with SpectralDistortionIndex score

    Example:
        >>> from torchmetrics import SpectralDistortionIndex
        >>> ms = torch.rand([16, 1, 16, 16])
        >>> fused = ms * 0.75
        >>> sdi = SpectralDistortionIndex()
        >>> sdi(ms, fused)
        tensor(0.9216)

    References:
    [1] Alparone, Luciano & Aiazzi, Bruno & Baronti, Stefano & Garzelli, Andrea & Nencini, Filippo & Selva, Massimo. (2008). Multispectral and Panchromatic Data Fusion Assessment Without Reference. ASPRS Journal of Photogrammetric Engineering and Remote Sensing. 74. 193-200. 10.14358/PERS.74.2.193.
    """

    ms: List[Tensor]
    fused: List[Tensor]
    higher_is_better: bool = True

    def __init__(
        self,
        reduction: Literal["elementwise_mean", "sum", "none"] = "elementwise_mean",
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )
        rank_zero_warn(
            "Metric `SpectralDistortionIndex` will save all targets and"
            " predictions in buffer. For large datasets this may lead"
            " to large memory footprint."
        )

        self.add_state("ms", default=[], dist_reduce_fx="cat")
        self.add_state("fused", default=[], dist_reduce_fx="cat")
        self.reduction = reduction

    def update(self, ms: Tensor, fused: Tensor, p: int = 1) -> None:  # type: ignore
        """Update state with ms and fused.

        Args:
            ms: Low resolution multispectral image
            fused: High resolution fused image
            p: Large spectral distortion (default: 1)
        """
        ms, fused, self.p = _d_lambda_update(ms, fused, p)
        self.ms.append(ms)
        self.fused.append(fused)

    def compute(self) -> Tensor:
        """Computes explained variance over state."""
        ms = dim_zero_cat(self.ms)
        fused = dim_zero_cat(self.fused)
        return _d_lambda_compute(ms, fused, self.p, self.reduction)
