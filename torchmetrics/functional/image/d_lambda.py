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

from typing import Tuple

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.image.uqi import universal_image_quality_index
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.distributed import reduce


def _d_lambda_update(preds: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    """Updates and returns variables required to compute Spectral Distortion Index. Checks for same shape and type
    of the input tensors.

    Args:
        preds: Low resolution multispectral image
        target: High resolution fused image
    """

    if preds.dtype != target.dtype:
        raise TypeError(
            f"Expected `ms` and `fused` to have the same data type. Got ms: {preds.dtype} and fused: {target.dtype}."
        )
    _check_same_shape(preds, target)
    if len(preds.shape) != 4:
        raise ValueError(
            f"Expected `preds` and `target` to have BxCxHxW shape. Got preds: {preds.shape} and target: {target.shape}."
        )
    return preds, target


def _d_lambda_compute(
    preds: Tensor,
    target: Tensor,
    p: int = 1,
    reduction: Literal["elementwise_mean", "sum", "none"] = "elementwise_mean",
) -> Tensor:
    """Computes Spectral Distortion Index.

    Args:
        preds: Low resolution multispectral image
        target: High resolution fused image
        p: a parameter to emphasize large spectral difference (default: 1)
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

    Example:
        >>> _ = torch.manual_seed(42)
        >>> preds = torch.rand([16, 3, 16, 16])
        >>> target = torch.rand([16, 3, 16, 16])
        >>> preds, target = _d_lambda_update(preds, target)
        >>> _d_lambda_compute(preds, target)
        tensor(0.0234)
    """
    if p <= 0:
        raise ValueError(f"Expected `p` to be a positive integer. Got p: {p}.")

    L = preds.shape[1]

    M1 = torch.zeros((L, L))
    M2 = torch.zeros((L, L))

    for k in range(L):
        for r in range(k, L):
            M1[k, r] = M1[r, k] = universal_image_quality_index(target[:, k : k + 1, :, :], target[:, r : r + 1, :, :])
            M2[k, r] = M2[r, k] = universal_image_quality_index(preds[:, k : k + 1, :, :], preds[:, r : r + 1, :, :])

    diff = torch.pow(torch.abs(M1 - M2), p)
    # Special case: when number of channels (L) is 1, there will be only one element in M1 and M2. Hence no need to sum.
    if L == 1:
        output = torch.pow(diff, (1.0 / p))
    else:
        output = torch.pow(1.0 / (L * (L - 1)) * torch.sum(diff), (1.0 / p))
    return reduce(output, reduction)


def spectral_distortion_index(
    preds: Tensor,
    target: Tensor,
    p: int = 1,
    reduction: Literal["elementwise_mean", "sum", "none"] = "elementwise_mean",
) -> Tensor:
    """Spectral Distortion Index.

    Args:
        preds: Low resolution multispectral image
        target: High resolution fused image
        p: Large spectral differences (default: 1)
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
        >>> from torchmetrics.functional import spectral_distortion_index
        >>> _ = torch.manual_seed(42)
        >>> preds = torch.rand([16, 3, 16, 16])
        >>> target = torch.rand([16, 3, 16, 16])
        >>> spectral_distortion_index(preds, target)
        tensor(0.0234)

    References:
        [1] Alparone, Luciano & Aiazzi, Bruno & Baronti, Stefano & Garzelli, Andrea & Nencini,
            Filippo & Selva, Massimo. (2008). Multispectral and Panchromatic Data Fusion
            Assessment Without Reference. ASPRS Journal of Photogrammetric Engineering
            and Remote Sensing. 74. 193-200. 10.14358/PERS.74.2.193.
    """
    preds, target = _d_lambda_update(preds, target)
    return _d_lambda_compute(preds, target, p, reduction)
