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
from typing import Optional

import torch
from torch import Tensor

from torchmetrics.utilities.checks import _check_same_shape


def _cosine_similarity_update(
    preds: Tensor,
    target: Tensor,
) -> tuple[Tensor, Tensor]:
    """Update and returns variables required to compute Cosine Similarity. Checks for same shape of input tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor

    """
    _check_same_shape(preds, target)
    if preds.ndim != 2:
        raise ValueError(
            "Expected input to cosine similarity to be 2D tensors of shape `[N,D]` where `N` is the number of samples"
            f" and `D` is the number of dimensions, but got tensor of shape {preds.shape}"
        )
    # Keep the input floating dtype. Casting everything to float32 silently truncated
    # float64 inputs, and a value that is representable in float64 (1e20) does not fit in a
    # float32 square, so the downcast on its own turned finite inputs into NaN. Half
    # precision stays promoted to float32, matching the previous behaviour, because its
    # norms are even less able to hold the intermediates.
    dtype = torch.promote_types(preds.dtype, target.dtype)
    if not dtype.is_floating_point or dtype.itemsize < 4:
        dtype = torch.get_default_dtype()
    return preds.to(dtype), target.to(dtype)


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
    # Cosine similarity does not change when either vector is scaled by a positive constant,
    # so divide out the largest magnitude first. Without that the squared norms overflow
    # (scale 1e20) or underflow (1e-30) even in float32, and the ratio came back as inf/inf or
    # 0/0 -- NaN for inputs that are perfectly finite. Scaling to a maximum of one keeps every
    # squared term at or below one and guarantees a norm of at least one for a nonzero vector,
    # while a zero vector still gives 0/0, exactly as before.
    tiny = torch.finfo(preds.dtype).tiny
    preds_scale = preds.abs().amax(dim=-1, keepdim=True).clamp_min(tiny)
    target_scale = target.abs().amax(dim=-1, keepdim=True).clamp_min(tiny)
    preds, target = preds / preds_scale, target / target_scale
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
