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
from typing import Any, List

from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.image.d_lambda import _spectral_distortion_index_compute, _spectral_distortion_index_update
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat


class SpectralDistortionIndex(Metric):
    """Computes Spectral Distortion Index (SpectralDistortionIndex_) also now as D_lambda is used to compare the
    spectral distortion between two images.

    Args:
        p: Large spectral differences
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.


    Example:
        >>> import torch
        >>> _ = torch.manual_seed(42)
        >>> from torchmetrics import SpectralDistortionIndex
        >>> preds = torch.rand([16, 3, 16, 16])
        >>> target = torch.rand([16, 3, 16, 16])
        >>> sdi = SpectralDistortionIndex()
        >>> sdi(preds, target)
        tensor(0.0234)

    References:
        [1] Alparone, Luciano & Aiazzi, Bruno & Baronti, Stefano & Garzelli, Andrea & Nencini,
        Filippo & Selva, Massimo. (2008). Multispectral and Panchromatic Data Fusion
        Assessment Without Reference. ASPRS Journal of Photogrammetric Engineering
        and Remote Sensing. 74. 193-200. 10.14358/PERS.74.2.193.
    """

    higher_is_better: bool = True
    is_differentiable: bool = True
    full_state_update: bool = False

    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
        self, p: int = 1, reduction: Literal["elementwise_mean", "sum", "none"] = "elementwise_mean", **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        rank_zero_warn(
            "Metric `SpectralDistortionIndex` will save all targets and"
            " predictions in buffer. For large datasets this may lead"
            " to large memory footprint."
        )

        if not isinstance(p, int) or p <= 0:
            raise ValueError(f"Expected `p` to be a positive integer. Got p: {p}.")
        self.p = p
        ALLOWED_REDUCTION = ("elementwise_mean", "sum", "none")
        if reduction not in ALLOWED_REDUCTION:
            raise ValueError(f"Expected argument `reduction` be one of {ALLOWED_REDUCTION} but got {reduction}")
        self.reduction = reduction
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with preds and target.

        Args:
            preds: Low resolution multispectral image
            target: High resolution fused image
        """
        preds, target = _spectral_distortion_index_update(preds, target)
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> Tensor:
        """Computes and returns spectral distortion index."""
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        return _spectral_distortion_index_compute(preds, target, self.p, self.reduction)
