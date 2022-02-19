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

import torch
from torch import Tensor
from typing import List, Tuple
from typing_extensions import Literal

from torchmetrics.functional.image.uqi import universal_image_quality_index
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.distributed import reduce


def _d_lambda_update(ms: Tensor, fused: Tensor, p: int) -> Tuple[Tensor, Tensor, Tensor]:
    """Updates and returns variables required to compute Spectral Distortion Index. Checks for same shape and
    type of the input tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
    """

    if ms.dtype != fused.dtype:
        raise TypeError(
            "Expected `ms` and `fused` to have the same data type."
            f" Got preds: {ms.dtype} and target: {fused.dtype}."
        )
    _check_same_shape(ms, fused)
    if len(ms.shape) != 4:
        raise ValueError(
            "Expected `ms` and `fused` to have BxCxHxW shape."
            f" Got preds: {ms.shape} and target: {fused.shape}."
        )
    if (p <= 0):
        raise ValueError(
            "Expected `p` to be a positive integer."
            f" Got p: {p}."
        )
    return (ms, fused, p)

def _d_lambda_compute(
    ms: Tensor,
    fused: Tensor,
    p: int,
    reduction: Literal["elementwise_mean", "sum", "none"] = "elementwise_mean",
) -> Tensor:
    """Computes Universal Image Quality Index.

    Args:
        preds: estimated image
        target: ground truth image
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

        data_range: Range of the image. If ``None``, it is determined from the image (max - min)

    Example:
        >>> preds = torch.rand([16, 1, 16, 16])
        >>> target = preds * 0.75
        >>> preds, target = _uqi_update(preds, target)
        >>> _uqi_compute(preds, target)
        tensor(0.9216)
    """

    # device = preds.device
    # channel = preds.size(1)
    # dtype = preds.dtype
    # kernel = _gaussian_kernel(channel, kernel_size, sigma, dtype, device)
    # pad_h = (kernel_size[0] - 1) // 2
    # pad_w = (kernel_size[1] - 1) // 2

    # preds = F.pad(preds, (pad_h, pad_h, pad_w, pad_w), mode="reflect")
    # target = F.pad(target, (pad_h, pad_h, pad_w, pad_w), mode="reflect")

    L = ms.shape[1]

    M1 = torch.zeros((L, L))
    M2 = torch.zeros((L, L))

    for l in range(L) :
        for r in range(l, L):
            print(fused.shape)
            M1[l, r] = M1[r, l] = universal_image_quality_index(fused[:, l, :, :], fused[:, r, :, :])
            M2[l, r] = M2[r, l] = universal_image_quality_index(ms[:, l, :, :], ms[:, r, :, :])
    
    diff = torch.pow(torch.abs(M1 - M2), p)
    output = torch.pow(1./(L*(L-1)) * torch.sum(diff), (1./p))
    return reduce(output, reduction)


def spectral_distortion_index(
    ms: Tensor,
    fused: Tensor,
    p: int = 1,
    reduction: Literal["elementwise_mean", "sum", "none"] = "elementwise_mean",
) -> Tensor:
    """Universal Image Quality Index.

    Args:
        preds: estimated image
        target: ground truth image
        kernel_size: size of the gaussian kernel (default: (11, 11))
        sigma: Standard deviation of the gaussian kernel (default: (1.5, 1.5))
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

        data_range: Range of the image. If ``None``, it is determined from the image (max - min)

    Return:
        Tensor with UniversalImageQualityIndex score

    Raises:
        TypeError:
            If ``preds`` and ``target`` don't have the same data type.
        ValueError:
            If ``preds`` and ``target`` don't have ``BxCxHxW shape``.
        ValueError:
            If the length of ``kernel_size`` or ``sigma`` is not ``2``.
        ValueError:
            If one of the elements of ``kernel_size`` is not an ``odd positive number``.
        ValueError:
            If one of the elements of ``sigma`` is not a ``positive number``.

    Example:
        >>> from torchmetrics.functional import universal_image_quality_index
        >>> preds = torch.rand([16, 1, 16, 16])
        >>> target = preds * 0.75
        >>> universal_image_quality_index(preds, target)
        tensor(0.9216)    
    """
    ms, fused, p = _d_lambda_update(ms, fused, p)
    return _d_lambda_compute(ms, fused, p, reduction)
