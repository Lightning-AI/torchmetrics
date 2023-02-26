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
from typing import Optional, Tuple

import torch
from torch import Tensor

from torchmetrics.utilities.checks import _check_same_shape


def _cosine_similarity_update(
    preds: Tensor,
    target: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Update and returns variables required to compute Cosine Similarity. Checks for same shape of input tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
    """
    _check_same_shape(preds, target)
    preds = preds.float()
    target = target.float()

    return preds, target


def _cosine_similarity_compute(preds: Tensor, target: Tensor, reduction: Optional[str] = "sum") -> Tensor:
    """Compute Cosine Similarity.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        reduction:
            The method of reducing along the batch dimension using sum, mean or taking the individual scores

    Example:
        >>> target = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
        >>> preds = torch.tensor([[1, 2, 3, 4], [-1, -2, -3, -4]])
        >>> preds, target = _cosine_similarity_update(preds, target)
        >>> _cosine_similarity_compute(preds, target, 'none')
        tensor([ 1.0000, -1.0000])
    """
    dot_product = (preds * target).sum(dim=-1)
    preds_norm = preds.norm(dim=-1)
    target_norm = target.norm(dim=-1)
    similarity = dot_product / (preds_norm * target_norm)
    reduction_mapping = {
        "sum": torch.sum,
        "mean": torch.mean,
        "none": lambda x: x,
        None: lambda x: x,
    }
    return reduction_mapping[reduction](similarity)  # type: ignore[operator]


def cosine_similarity(preds: Tensor, target: Tensor, reduction: Optional[str] = "sum") -> Tensor:
    r"""Compute the `Cosine Similarity`_.

    .. math::
        cos_{sim}(x,y) = \frac{x \cdot y}{||x|| \cdot ||y||} =
        \frac{\sum_{i=1}^n x_i y_i}{\sqrt{\sum_{i=1}^n x_i^2}\sqrt{\sum_{i=1}^n y_i^2}}

    where :math:`y` is a tensor of target values, and :math:`x` is a tensor of predictions.

    Args:
        preds: Predicted tensor with shape ``(N,d)``
        target: Ground truth tensor with shape ``(N,d)``
        reduction:
            The method of reducing along the batch dimension using sum, mean or taking the individual scores

    Example:
        >>> from torchmetrics.functional.regression import cosine_similarity
        >>> target = torch.tensor([[1, 2, 3, 4],
        ...                        [1, 2, 3, 4]])
        >>> preds = torch.tensor([[1, 2, 3, 4],
        ...                       [-1, -2, -3, -4]])
        >>> cosine_similarity(preds, target, 'none')
        tensor([ 1.0000, -1.0000])
    """
    preds, target = _cosine_similarity_update(preds, target)
    return _cosine_similarity_compute(preds, target, reduction)
