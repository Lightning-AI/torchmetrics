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
from typing import Any, List, Optional, Sequence

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.image.d_lambda import _d_lambda_update, _d_lambda_compute
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat


class SpectralDistortionIndex(Metric):
    """Computes Universal Image Quality Index (UniversalImageQualityIndex_).

    Args:
        kernel_size: size of the gaussian kernel
        sigma: Standard deviation of the gaussian kernel
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

        data_range: Range of the image. If ``None``, it is determined from the image (max - min)

    Return:
        Tensor with UniversalImageQualityIndex score

    Example:
        >>> from torchmetrics import UniversalImageQualityIndex
        >>> preds = torch.rand([16, 1, 16, 16])
        >>> target = preds * 0.75
        >>> uqi = UniversalImageQualityIndex()
        >>> uqi(preds, target)
        tensor(0.9216)
    """

    ms: List[Tensor]
    fused: List[Tensor]
    p: int
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
        self.p = 1
        self.reduction = reduction

    def update(self, ms: Tensor, fused: Tensor, p: int) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        ms, fused, p = _d_lambda_update(ms, fused, p)
        self.ms.append(ms)
        self.fused.append(fused)
        self.p = p

    def compute(self) -> Tensor:
        """Computes explained variance over state."""
        ms = dim_zero_cat(self.ms)
        fused = dim_zero_cat(self.fused)
        return _d_lambda_compute(ms, fused, self.p, self.reduction)
