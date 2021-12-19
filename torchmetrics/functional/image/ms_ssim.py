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
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.image.ssim import _ssim_compute, _ssim_update


def _get_normalized_sim_and_cs(
    preds: Tensor,
    target: Tensor,
    kernel_size: Sequence[int] = (11, 11),
    sigma: Sequence[float] = (1.5, 1.5),
    reduction: str = "elementwise_mean",
    data_range: Optional[float] = None,
    k1: float = 0.01,
    k2: float = 0.03,
    normalize: Optional[Literal["relu", "simple"]] = None,
) -> Tuple[Tensor, Tensor]:
    sim, contrast_sensitivity = _ssim_compute(
        preds, target, kernel_size, sigma, reduction, data_range, k1, k2, return_contrast_sensitivity=True
    )
    if normalize == "relu":
        sim = torch.relu(sim)
        contrast_sensitivity = torch.relu(contrast_sensitivity)
    return sim, contrast_sensitivity


def _ms_ssim_compute(
    preds: Tensor,
    target: Tensor,
    kernel_size: Sequence[int] = (11, 11),
    sigma: Sequence[float] = (1.5, 1.5),
    reduction: str = "elementwise_mean",
    data_range: Optional[float] = None,
    k1: float = 0.01,
    k2: float = 0.03,
    betas: Union[Tuple[float, float, float, float, float], Tuple[float, ...]] = (
        0.0448,
        0.2856,
        0.3001,
        0.2363,
        0.1333,
    ),
    normalize: Optional[Literal["relu", "simple"]] = None,
) -> Tensor:
    """Computes Multi-Scale Structual Similarity Index Measure. Adapted from: https://github.com/jorge-
    pessoa/pytorch-msssim/blob/master/pytorch_msssim/__init__.py.

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
        k1: Parameter of SSIM.
        k2: Parameter of SSIM.
    """
    sim_list: List[Tensor] = []
    cs_list: List[Tensor] = []

    if preds.size()[-1] < 2 ** len(betas) or preds.size()[-2] < 2 ** len(betas):
        raise ValueError(
            f"For a given number of `betas` parameters {len(betas)}, the image height and width dimensions must be "
            f"larger than or equal to {2 ** len(betas)}."
        )

    _betas_div = max(1, (len(betas) - 1)) ** 2
    if preds.size()[-2] // _betas_div <= kernel_size[0] - 1:
        raise ValueError(
            f"For a given number of `betas` parameters {len(betas)} and kernel size {kernel_size[0]}, the image height "
            f"must be larger than {(kernel_size[0] - 1) * _betas_div}."
        )
    if preds.size()[-1] // _betas_div <= kernel_size[1] - 1:
        raise ValueError(
            f"For a given number of `betas` parameters {len(betas)} and kernel size {kernel_size[1]}, the image width "
            f"must be larger than {(kernel_size[1] - 1) * _betas_div}."
        )

    for _ in range(len(betas)):
        sim, contrast_sensitivity = _get_normalized_sim_and_cs(
            preds, target, kernel_size, sigma, reduction, data_range, k1, k2, normalize
        )
        sim_list.append(sim)
        cs_list.append(contrast_sensitivity)
        preds = F.avg_pool2d(preds, (2, 2))
        target = F.avg_pool2d(target, (2, 2))

    sim_stack = torch.stack(sim_list)
    cs_stack = torch.stack(cs_list)

    if normalize == "simple":
        sim_stack = (sim_stack + 1) / 2
        cs_stack = (cs_stack + 1) / 2

    sim_stack = sim_stack ** torch.tensor(betas)
    cs_stack = cs_stack ** torch.tensor(betas)
    return torch.prod(cs_stack[:-1]) * sim_stack[-1]


def ms_ssim(
    preds: Tensor,
    target: Tensor,
    kernel_size: Sequence[int] = (11, 11),
    sigma: Sequence[float] = (1.5, 1.5),
    reduction: str = "elementwise_mean",
    data_range: Optional[float] = None,
    k1: float = 0.01,
    k2: float = 0.03,
    betas: Tuple[float, float, float, float, float] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
    normalize: Optional[Literal["relu", "simple"]] = None,
) -> Tensor:
    """Computes `MultiScaleSSIM`_, Multi-scale Structual Similarity Index Measure, which is a generalization of
    Structual Similarity Index Measure by incorporating image details at different resolution scores.

    Args:
        preds: Predictions from model of shape `[N, C, H, W]`
        target: Ground truth values of shape `[N, C, H, W]`
        kernel_size: size of the gaussian kernel (default: (11, 11))
        sigma: Standard deviation of the gaussian kernel (default: (1.5, 1.5))
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

        data_range: Range of the image. If ``None``, it is determined from the image (max - min)
        k1: Parameter of SSIM.
        k2: Parameter of SSIM.
        betas: Exponent parameters for individual similarities and contrastive sensitivies returned by different image
        resolutions.
        normalize: When MS-SSIM loss is used for training, it is desirable to use normalizes to improve the training
        stability. This `normalize` argument is out of scope of the original implementation [1], and it is adapted from
        https://github.com/jorge-pessoa/pytorch-msssim instead.

    Return:
        Tensor with Multi-Scale SSIM score

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
        >>> from torchmetrics.functional import ms_ssim
        >>> preds = torch.rand([1, 1, 256, 256], generator=torch.manual_seed(42))
        >>> target = preds * 0.75
        >>> ms_ssim(preds, target)
        tensor(0.9558)

    References:
    [1] Multi-Scale Structural Similarity For Image Quality Assessment by Zhou Wang, Eero P. Simoncelli and Alan C.
    Bovik `MultiScaleSSIM`_
    """
    if not isinstance(betas, tuple):
        raise ValueError("Argument `betas` is expected to be of a type tuple.")
    if isinstance(betas, tuple) and not all(isinstance(beta, float) for beta in betas):
        raise ValueError("Argument `betas` is expected to be a tuple of floats.")
    if normalize and normalize not in ("relu", "simple"):
        raise ValueError("Argument `normalize` to be expected either `None` or one of 'relu' or 'simple'")

    preds, target = _ssim_update(preds, target)
    return _ms_ssim_compute(preds, target, kernel_size, sigma, reduction, data_range, k1, k2, betas, normalize)
