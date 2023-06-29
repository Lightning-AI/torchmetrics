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
from typing import Optional, Tuple, Union

from torch import Tensor
from typing_extensions import Literal


def _total_variation_update(img: Tensor) -> Tuple[Tensor, int]:
    """Compute total variation statistics on current batch."""
    if img.ndim != 4:
        raise RuntimeError(f"Expected input `img` to be an 4D tensor, but got {img.shape}")
    diff1 = img[..., 1:, :] - img[..., :-1, :]
    diff2 = img[..., :, 1:] - img[..., :, :-1]

    res1 = diff1.abs().sum([1, 2, 3])
    res2 = diff2.abs().sum([1, 2, 3])
    score = res1 + res2
    return score, img.shape[0]


def _total_variation_compute(
    score: Tensor, num_elements: Union[int, Tensor], reduction: Optional[Literal["mean", "sum", "none"]]
) -> Tensor:
    """Compute final total variation score."""
    if reduction == "mean":
        return score.sum() / num_elements
    if reduction == "sum":
        return score.sum()
    if reduction is None or reduction == "none":
        return score
    raise ValueError("Expected argument `reduction` to either be 'sum', 'mean', 'none' or None")


def total_variation(img: Tensor, reduction: Optional[Literal["mean", "sum", "none"]] = "sum") -> Tensor:
    """Compute total variation loss.

    Args:
        img: A `Tensor` of shape `(N, C, H, W)` consisting of images
        reduction: a method to reduce metric score over samples.

            - ``'mean'``: takes the mean over samples
            - ``'sum'``: takes the sum over samples
            - ``None`` or ``'none'``: return the score per sample

    Returns:
        A loss scalar value containing the total variation

    Raises:
        ValueError:
            If ``reduction`` is not one of ``'sum'``, ``'mean'``, ``'none'`` or ``None``
        RuntimeError:
            If ``img`` is not 4D tensor

    Example:
        >>> import torch
        >>> from torchmetrics.functional.image import total_variation
        >>> _ = torch.manual_seed(42)
        >>> img = torch.rand(5, 3, 28, 28)
        >>> total_variation(img)
        tensor(7546.8018)
    """
    # code adapted from:
    # from kornia.losses import total_variation as kornia_total_variation
    score, num_elements = _total_variation_update(img)
    return _total_variation_compute(score, num_elements, reduction)
