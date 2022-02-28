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

from typing import List, Tuple

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.image.uqi import universal_image_quality_index
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.distributed import reduce


def _d_lambda_update(ms: Tensor, fused: Tensor) -> Tuple[Tensor, Tensor]:
    """Updates and returns variables required to compute Spectral Distortion Index. Checks for same shape and type
    of the input tensors.

    Args:
        ms: Low resolution multispectral image
        fused: High resolution fused image
    """

    if ms.dtype != fused.dtype:
        raise TypeError(
            "Expected `ms` and `fused` to have the same data type." f" Got ms: {ms.dtype} and fused: {fused.dtype}."
        )
    _check_same_shape(ms, fused)
    if len(ms.shape) != 4:
        raise ValueError(
            "Expected `ms` and `fused` to have BxCxHxW shape." f" Got ms: {ms.shape} and fused: {fused.shape}."
        )
    return ms, fused


def _d_lambda_compute(
    ms: Tensor,
    fused: Tensor,
    p: int = 1,
    reduction: Literal["elementwise_mean", "sum", "none"] = "elementwise_mean",
) -> Tensor:
    """Computes Spectral Distortion Index.

    Args:
        ms: Low resolution multispectral image
        fused: High resolution fused image
        p: a parameter to emphasize large spectral difference (default: 1)
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

    Example:
        >>> ms = torch.rand([16, 3, 16, 16])
        >>> fused = ms * 0.75
        >>> ms, target = _d_lambda_update(ms, fused)
        >>> _d_lambda_compute(ms, fused)
        tensor(0.9216)

    References:
    [1] Alparone, Luciano & Aiazzi, Bruno & Baronti, Stefano & Garzelli, Andrea & Nencini, Filippo & Selva, Massimo. (2008). Multispectral and Panchromatic Data Fusion Assessment Without Reference. ASPRS Journal of Photogrammetric Engineering and Remote Sensing. 74. 193-200. 10.14358/PERS.74.2.193.
    """
    if p <= 0:
        raise ValueError(f"Expected `p` to be a positive integer. Got p: {p}.")

    L = ms.shape[1]

    M1 = torch.zeros((L, L))
    M2 = torch.zeros((L, L))

    for l in range(L):
        for r in range(l, L):
            M1[l, r] = M1[r, l] = universal_image_quality_index(fused[:, l : l + 1, :, :], fused[:, r : r + 1, :, :])
            M2[l, r] = M2[r, l] = universal_image_quality_index(ms[:, l : l + 1, :, :], ms[:, r : r + 1, :, :])

    diff = torch.pow(torch.abs(M1 - M2), p)
    # Special case: when number of channels (L) is 1, there will be only one element in M1 and M2. Hence no need to sum.
    if L == 1:
        output = torch.pow(diff, (1.0 / p))
    else:
        output = torch.pow(1.0 / (L * (L - 1)) * torch.sum(diff), (1.0 / p))
    return reduce(output, reduction)


def spectral_distortion_index(
    ms: Tensor,
    fused: Tensor,
    p: int = 1,
    reduction: Literal["elementwise_mean", "sum", "none"] = "elementwise_mean",
) -> Tensor:
    """Spectral Distortion Index.

    Args:
        ms: Low resolution multispectral image
        fused: High resolution fused image
        p: Large spectral differences (default: 1)
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

    Return:
        Tensor with SpectralDistortionIndex score

    Raises:
        TypeError:
            If ``ms`` and ``fused`` don't have the same data type.
        ValueError:
            If ``ms`` and ``fused`` don't have ``BxCxHxW shape``.
        ValueError:
            If ``p`` is not a positive integer.

    Example:
        >>> from torchmetrics.functional import spectral_distortion_index
        >>> ms = torch.rand([16, 3, 16, 16])
        >>> fused = ms * 0.75
        >>> spectral_distortion_index(ms, fused)
        tensor(0.9216)
    """
    ms, fused = _d_lambda_update(ms, fused)
    return _d_lambda_compute(ms, fused, p, reduction)
