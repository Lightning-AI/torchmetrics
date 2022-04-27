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
from torch.nn import functional as F
from typing_extensions import Literal

from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.distributed import reduce


def _scc_update(preds: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    """Updates and returns variables required to compute Spatial Correlation Coefficient.

    Checks for same shape and
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


def _scc_compute(
    preds: Tensor,
    target: Tensor,
    kernel_size: Sequence[int] = (9, 9),
    reduction: Optional[Literal["elementwise_mean", "sum", "none"]] = "elementwise_mean",
) -> Union[Tensor, Tuple[Tensor, Tensor]]:

    """Args:
        preds: estimated image
        target: ground truth image
        kernel_size: size of the Uniform kernel (default: (9, 9))

        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied


    Return:
        Tensor with Spatial Correlation Coefficient score

    Raises:
        TypeError:
            If ``preds`` and ``target`` don't have the same data type.
        ValueError:
            If ``preds`` and ``target`` don't have ``BxCxHxW shape``.
        ValueError:
            If the length of ``kernel_size`` is not ``2``.
        ValueError:
            If one of the elements of ``kernel_size`` is not an ``odd positive number``."""

    if len(kernel_size) != 2:
        raise ValueError(
            "Expected `kernel_size` and `sigma` to have the length of two." f" Got kernel_size: {len(kernel_size)}."
        )

    if any(x % 2 == 0 or x <= 0 for x in kernel_size):
        raise ValueError(f"Expected `kernel_size` to have odd positive number. Got {kernel_size}.")

    batch_size = preds.shape[0]

    classes = preds.shape[1]

    coefs = torch.zeros((batch_size, classes, preds.shape[2], preds.shape[3]))

    kernel = torch.div(
        torch.ones(1, 1, kernel_size[0], kernel_size[1], dtype=preds.dtype, device=preds.device),
        kernel_size[0] * kernel_size[1],
    )

    for i in range(classes):

        mu1, mu2 = F.conv2d(preds[:, i, :, :].unsqueeze(1), kernel, padding="same"), F.conv2d(
            target[:, i, :, :].unsqueeze(1), kernel, padding="same"
        )

        preds_sum_sq, target_sum_sq, preds_target_sum_mul = mu1 * mu1, mu2 * mu2, mu1 * mu2

        outputs_1 = F.conv2d(
            preds[:, i, :, :].unsqueeze(1) * preds[:, i, :, :].unsqueeze(1),
            kernel,
            padding="same",
        )
        outputs_1 -= preds_sum_sq

        outputs_2 = F.conv2d(
            target[:, i, :, :].unsqueeze(1) * target[:, i, :, :].unsqueeze(1),
            kernel,
            padding="same",
        )
        outputs_2 -= target_sum_sq

        outputs_3 = F.conv2d(
            preds[:, i, :, :].unsqueeze(1) * target[:, i, :, :].unsqueeze(1),
            kernel,
            padding="same",
        )
        outputs_3 -= preds_target_sum_mul

        sigma_preds_sq, sigma_target_sq, sigma_preds_target = outputs_1, outputs_2, outputs_3

        sigma_preds_sq[sigma_preds_sq < 0] = 0
        sigma_target_sq[sigma_target_sq < 0] = 0

        den = torch.sqrt(sigma_preds_sq) * torch.sqrt(sigma_target_sq)

        idx = den == 0

        den[den == 0] = 1

        scc = sigma_preds_target / den

        scc[idx] = 0

        coefs[:, i, :, :] = scc.squeeze(1)

    batch_score = []
    for i in range(scc.shape[0]):
        batch_score.append(torch.mean(scc[i, :, :, :]))

    final_batch_score = torch.as_tensor(batch_score)

    final_batch_score = reduce(final_batch_score, reduction=reduction)

    return final_batch_score


def spatial_correlation_coefficient(
    preds: Tensor,
    target: Tensor,
    kernel_size: Sequence[int] = (9, 9),
    reduction: Optional[Literal["elementwise_mean", "sum", "none"]] = "elementwise_mean",
) -> Tensor:
    """Spatial Correlation Coefficient.

    Args:
        preds: estimated image
        target: ground truth image
        kernel_size: size of the Uniform kernel (default: (9, 9))

        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

    Return:
        Tensor with Spatial Correlation Coefficient score

    Raises:
        TypeError:
            If ``preds`` and ``target`` don't have the same data type.
        ValueError:
            If ``preds`` and ``target`` don't have ``BxCxHxW shape``.
        ValueError:
            If the length of ``kernel_size``  is not ``2``.
        ValueError:
            If one of the elements of ``kernel_size`` is not an ``odd positive number``.

    Example:
        >>> from torchmetrics.functional.image.scc import spatial_correlation_coefficient
        >>> preds = torch.ones([16, 3, 16, 16])
        >>> target = torch.ones([16, 3, 16, 16])
        >>> spatial_correlation_coefficient(preds, target)
        tensor(1.)
    """

    preds, target = _scc_update(preds, target)

    return _scc_compute(preds, target, kernel_size, reduction)
