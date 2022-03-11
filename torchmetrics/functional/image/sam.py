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
from typing import Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.regression.cosine_similarity import cosine_similarity
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.distributed import reduce


def _sam_update(preds: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    """Updates and returns variables required to compute Spectral Angle Mapper. Checks for same shape and
    type of the input tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
    """

    if preds.dtype != target.dtype:
        raise TypeError(
            "Expected `preds` and `target` to have the same data type."
            f" Got preds: {preds.dtype} and target: {target.dtype}."
        )
    _check_same_shape(preds, target)
    if len(preds.shape) != 4:
        raise ValueError(
            "Expected `preds` and `target` to have BxCxHxW shape."
            f" Got preds: {preds.shape} and target: {target.shape}."
        )
    return preds, target


def _sam_compute(
        preds: Tensor,
        target: Tensor,
        reduction: Literal["elementwise_mean", "sum", "none"] = "elementwise_mean",
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Computes Spectral Angle Mapper.

    Args:
        preds: estimated image
        target: ground truth image
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

    Example:
        >>> preds = torch.rand([16, 1, 16, 16])
        >>> target = torch.rand([16, 1, 16, 16])
        >>> preds, target = _sam_update(preds, target)
        >>> _sam_compute(preds, target)
        tensor(0.7542)
    """
    B, C, H, W = preds.shape
    preds = preds.reshape(B, C, H * W)
    target = target.reshape(B, C, H * W)
    sam_score = torch.clip(cosine_similarity(preds, target, reduction="none"), -1, 1).arccos()
    return reduce(sam_score, reduction)


def universal_spectral_angle_mapper(
    preds: Tensor,
    target: Tensor,
    reduction: Literal["elementwise_mean", "sum", "none"] = "elementwise_mean",
) -> Tensor:
    """Universal Spectral Angle Mapper.

    Args:
        preds: estimated image
        target: ground truth image
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

    Return:
        Tensor with Spectral Angle Mapper score

    Raises:
        TypeError:
            If ``preds`` and ``target`` don't have the same data type.
        ValueError:
            If ``preds`` and ``target`` don't have ``BxCxHxW shape``.

    Example:
        >>> from torchmetrics.functional import universal_spectral_angle_mapper
        >>> preds = torch.rand([16, 1, 16, 16])
        >>> target = torch.rand([16, 1, 16, 16])
        >>> universal_spectral_angle_mapper(preds, target)
        tensor(0.7542)

    References: Roberta H. Yuhas, Alexander F. H. Goetz and Joe W. Boardman, "Discrimination among semi-arid
    landscape endmembers using the Spectral Angle Mapper (SAM) algorithm" in PL, Summaries of the Third Annual JPL
    Airborne Geoscience Workshop, vol. 1, June 1, 1992.
    """
    preds, target = _sam_update(preds, target)
    return _sam_compute(preds, target, reduction)