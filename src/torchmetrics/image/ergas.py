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

from typing import Any, List, Union

from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.image.ergas import _ergas_compute, _ergas_update
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat


class ErrorRelativeGlobalDimensionlessSynthesis(Metric):
    """Relative dimensionless global error synthesis (ERGAS) is used to calculate the accuracy of Pan sharpened
    image considering normalized average error of each band of the result image
    (ErrorRelativeGlobalDimensionlessSynthesis).

    Args:
        ratio: ratio of high resolution to low resolution
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Return:
        Tensor with ErrorRelativeGlobalDimensionlessSynthesis score

    Example:
        >>> import torch
        >>> from torchmetrics import ErrorRelativeGlobalDimensionlessSynthesis
        >>> preds = torch.rand([16, 1, 16, 16], generator=torch.manual_seed(42))
        >>> target = preds * 0.75
        >>> ergas = ErrorRelativeGlobalDimensionlessSynthesis()
        >>> torch.round(ergas(preds, target))
        tensor(154.)

    References:
        [1] Qian Du; Nicholas H. Younan; Roger King; Vijay P. Shah, "On the Performance Evaluation of
        Pan-Sharpening Techniques" in IEEE Geoscience and Remote Sensing Letters, vol. 4, no. 4, pp. 518-522,
        15 October 2007, doi: 10.1109/LGRS.2007.896328.
    """

    higher_is_better: bool = False
    is_differentiable: bool = True
    full_state_update: bool = False

    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
        self,
        ratio: Union[int, float] = 4,
        reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        rank_zero_warn(
            "Metric `UniversalImageQualityIndex` will save all targets and"
            " predictions in buffer. For large datasets this may lead"
            " to large memory footprint."
        )

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.ratio = ratio
        self.reduction = reduction

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        preds, target = _ergas_update(preds, target)
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> Tensor:
        """Computes explained variance over state."""
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        return _ergas_compute(preds, target, self.ratio, self.reduction)
