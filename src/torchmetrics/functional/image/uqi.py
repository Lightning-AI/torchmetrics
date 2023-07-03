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
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812
from typing_extensions import Literal

from torchmetrics.functional.image.helper import _gaussian_kernel_2d
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.distributed import reduce


def _uqi_update(preds: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    """Update and returns variables required to compute Universal Image Quality Index.

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


def _uqi_compute(
    preds: Tensor,
    target: Tensor,
    kernel_size: Sequence[int] = (11, 11),
    sigma: Sequence[float] = (1.5, 1.5),
    reduction: Optional[Literal["elementwise_mean", "sum", "none"]] = "elementwise_mean",
) -> Tensor:
    """Compute Universal Image Quality Index.

    Args:
        preds: estimated image
        target: ground truth image
        kernel_size: size of the gaussian kernel
        sigma: Standard deviation of the gaussian kernel
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

    Example:
        >>> preds = torch.rand([16, 1, 16, 16])
        >>> target = preds * 0.75
        >>> preds, target = _uqi_update(preds, target)
        >>> _uqi_compute(preds, target)
        tensor(0.9216)
    """
    if len(kernel_size) != 2 or len(sigma) != 2:
        raise ValueError(
            "Expected `kernel_size` and `sigma` to have the length of two."
            f" Got kernel_size: {len(kernel_size)} and sigma: {len(sigma)}."
        )

    if any(x % 2 == 0 or x <= 0 for x in kernel_size):
        raise ValueError(f"Expected `kernel_size` to have odd positive number. Got {kernel_size}.")

    if any(y <= 0 for y in sigma):
        raise ValueError(f"Expected `sigma` to have positive number. Got {sigma}.")

    device = preds.device
    channel = preds.size(1)
    dtype = preds.dtype
    kernel = _gaussian_kernel_2d(channel, kernel_size, sigma, dtype, device)
    pad_h = (kernel_size[0] - 1) // 2
    pad_w = (kernel_size[1] - 1) // 2

    preds = F.pad(preds, (pad_h, pad_h, pad_w, pad_w), mode="reflect")
    target = F.pad(target, (pad_h, pad_h, pad_w, pad_w), mode="reflect")

    input_list = torch.cat((preds, target, preds * preds, target * target, preds * target))  # (5 * B, C, H, W)
    outputs = F.conv2d(input_list, kernel, groups=channel)
    output_list = outputs.split(preds.shape[0])

    mu_pred_sq = output_list[0].pow(2)
    mu_target_sq = output_list[1].pow(2)
    mu_pred_target = output_list[0] * output_list[1]

    sigma_pred_sq = output_list[2] - mu_pred_sq
    sigma_target_sq = output_list[3] - mu_target_sq
    sigma_pred_target = output_list[4] - mu_pred_target

    upper = 2 * sigma_pred_target
    lower = sigma_pred_sq + sigma_target_sq + torch.finfo(sigma_pred_sq.dtype).eps

    uqi_idx = ((2 * mu_pred_target) * upper) / ((mu_pred_sq + mu_target_sq) * lower)
    uqi_idx = uqi_idx[..., pad_h:-pad_h, pad_w:-pad_w]

    return reduce(uqi_idx, reduction)


def universal_image_quality_index(
    preds: Tensor,
    target: Tensor,
    kernel_size: Sequence[int] = (11, 11),
    sigma: Sequence[float] = (1.5, 1.5),
    reduction: Optional[Literal["elementwise_mean", "sum", "none"]] = "elementwise_mean",
) -> Tensor:
    """Universal Image Quality Index.

    Args:
        preds: estimated image
        target: ground truth image
        kernel_size: size of the gaussian kernel
        sigma: Standard deviation of the gaussian kernel
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

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
        >>> from torchmetrics.functional.image import universal_image_quality_index
        >>> preds = torch.rand([16, 1, 16, 16])
        >>> target = preds * 0.75
        >>> universal_image_quality_index(preds, target)
        tensor(0.9216)

    References:
        [1] Zhou Wang and A. C. Bovik, "A universal image quality index," in IEEE Signal Processing Letters, vol. 9,
        no. 3, pp. 81-84, March 2002, doi: 10.1109/97.995823.

        [2] Zhou Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli, "Image quality assessment: from error visibility
        to structural similarity," in IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600-612, April 2004,
        doi: 10.1109/TIP.2003.819861.
    """
    preds, target = _uqi_update(preds, target)
    return _uqi_compute(preds, target, kernel_size, sigma, reduction)
