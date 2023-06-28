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

from typing import Tuple

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.image.uqi import universal_image_quality_index
from torchmetrics.utilities.distributed import reduce


def _spectral_distortion_index_update(preds: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    """Update and returns variables required to compute Spectral Distortion Index.

    Args:
        preds: Low resolution multispectral image
        target: High resolution fused image
    """
    if preds.dtype != target.dtype:
        raise TypeError(
            f"Expected `ms` and `fused` to have the same data type. Got ms: {preds.dtype} and fused: {target.dtype}."
        )
    if len(preds.shape) != 4:
        raise ValueError(
            f"Expected `preds` and `target` to have BxCxHxW shape. Got preds: {preds.shape} and target: {target.shape}."
        )
    if preds.shape[:2] != target.shape[:2]:
        raise ValueError(
            "Expected `preds` and `target` to have same batch and channel sizes."
            f"Got preds: {preds.shape} and target: {target.shape}."
        )
    return preds, target


def _spectral_distortion_index_compute(
    preds: Tensor,
    target: Tensor,
    p: int = 1,
    reduction: Literal["elementwise_mean", "sum", "none"] = "elementwise_mean",
) -> Tensor:
    """Compute Spectral Distortion Index (SpectralDistortionIndex_).

    Args:
        preds: Low resolution multispectral image
        target: High resolution fused image
        p: a parameter to emphasize large spectral difference
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

    Example:
        >>> _ = torch.manual_seed(42)
        >>> preds = torch.rand([16, 3, 16, 16])
        >>> target = torch.rand([16, 3, 16, 16])
        >>> preds, target = _spectral_distortion_index_update(preds, target)
        >>> _spectral_distortion_index_compute(preds, target)
        tensor(0.0234)
    """
    length = preds.shape[1]

    m1 = torch.zeros((length, length), device=preds.device)
    m2 = torch.zeros((length, length), device=preds.device)

    for k in range(length):
        num = length - (k + 1)
        if num == 0:
            continue
        stack1 = target[:, k : k + 1, :, :].repeat(num, 1, 1, 1)
        stack2 = torch.cat([target[:, r : r + 1, :, :] for r in range(k + 1, length)], dim=0)
        score = [
            s.mean() for s in universal_image_quality_index(stack1, stack2, reduction="none").split(preds.shape[0])
        ]
        m1[k, k + 1 :] = torch.stack(score, 0)

        stack1 = preds[:, k : k + 1, :, :].repeat(num, 1, 1, 1)
        stack2 = torch.cat([preds[:, r : r + 1, :, :] for r in range(k + 1, length)], dim=0)
        score = [
            s.mean() for s in universal_image_quality_index(stack1, stack2, reduction="none").split(preds.shape[0])
        ]
        m2[k, k + 1 :] = torch.stack(score, 0)
    m1 = m1 + m1.T
    m2 = m2 + m2.T

    diff = torch.pow(torch.abs(m1 - m2), p)
    # Special case: when number of channels (L) is 1, there will be only one element in M1 and M2. Hence no need to sum.
    if length == 1:
        output = torch.pow(diff, (1.0 / p))
    else:
        output = torch.pow(1.0 / (length * (length - 1)) * torch.sum(diff), (1.0 / p))
    return reduce(output, reduction)


def spectral_distortion_index(
    preds: Tensor,
    target: Tensor,
    p: int = 1,
    reduction: Literal["elementwise_mean", "sum", "none"] = "elementwise_mean",
) -> Tensor:
    """Calculate `Spectral Distortion Index`_ (SpectralDistortionIndex_) also known as D_lambda.

    Metric is used to compare the spectral distortion between two images.

    Args:
        preds: Low resolution multispectral image
        target: High resolution fused image
        p: Large spectral differences
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

    Return:
        Tensor with SpectralDistortionIndex score

    Raises:
        TypeError:
            If ``preds`` and ``target`` don't have the same data type.
        ValueError:
            If ``preds`` and ``target`` don't have ``BxCxHxW shape``.
        ValueError:
            If ``p`` is not a positive integer.

    Example:
        >>> from torchmetrics.functional.image import spectral_distortion_index
        >>> _ = torch.manual_seed(42)
        >>> preds = torch.rand([16, 3, 16, 16])
        >>> target = torch.rand([16, 3, 16, 16])
        >>> spectral_distortion_index(preds, target)
        tensor(0.0234)
    """
    if not isinstance(p, int) or p <= 0:
        raise ValueError(f"Expected `p` to be a positive integer. Got p: {p}.")
    preds, target = _spectral_distortion_index_update(preds, target)
    return _spectral_distortion_index_compute(preds, target, p, reduction)
