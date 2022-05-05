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
from torch import Tensor
from torch.nn import functional as F
from typing_extensions import Literal

from torchmetrics.functional.image.helper import _gaussian_kernel_2d, _gaussian_kernel_3d, _reflection_pad_3d
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.distributed import reduce


def _ssim_update(preds: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    """Updates and returns variables required to compute Structural Similarity Index Measure. Checks for same shape
    and type of the input tensors.

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
    if len(preds.shape) not in (4, 5):
        raise ValueError(
            "Expected `preds` and `target` to have BxCxHxW or BxCxDxHxW shape."
            f" Got preds: {preds.shape} and target: {target.shape}."
        )
    return preds, target


def _ssim_compute(
    preds: Tensor,
    target: Tensor,
    gaussian_kernel: bool = True,
    sigma: Union[float, Sequence[float]] = 1.5,
    kernel_size: Union[int, Sequence[int]] = 11,
    reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
    data_range: Optional[float] = None,
    k1: float = 0.01,
    k2: float = 0.03,
    return_full_image: bool = False,
    return_contrast_sensitivity: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Computes Structual Similarity Index Measure.

    Args:
        preds: estimated image
        target: ground truth image
        gaussian_kernel: If true (default), a gaussian kernel is used, if false a uniform kernel is used
        sigma: Standard deviation of the gaussian kernel, anisotropic kernels are possible.
            Ignored if a uniform kernel is used
        kernel_size: the size of the uniform kernel, anisotropic kernels are possible.
            Ignored if a Gaussian kernel is used
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

        data_range: Range of the image. If ``None``, it is determined from the image (max - min)
        k1: Parameter of SSIM.
        k2: Parameter of SSIM.
        return_full_image: If true, the full ``ssim`` image is returned as a second argument.
            Mutually exlusive with ``return_contrast_sensitivity``
        return_contrast_sensitivity: If true, the contrast term is returned as a second argument.
            The luminance term can be obtained with luminance=ssim/contrast
            Mutually exclusive with ``return_full_image``

    Example:
        >>> preds = torch.rand([16, 1, 16, 16])
        >>> target = preds * 0.75
        >>> preds, target = _ssim_update(preds, target)
        >>> _ssim_compute(preds, target)
        tensor(0.9219)
    """
    is_3d = len(preds.shape) == 5

    if not isinstance(kernel_size, Sequence):
        kernel_size = 3 * [kernel_size] if is_3d else 2 * [kernel_size]
    if not isinstance(sigma, Sequence):
        sigma = 3 * [sigma] if is_3d else 2 * [sigma]

    if len(kernel_size) != len(target.shape) - 2:
        raise ValueError(
            f"`kernel_size` has dimension {len(kernel_size)}, but expected to be two less that target dimensionality,"
            f" which is: {len(target.shape)}"
        )
    if len(kernel_size) not in (2, 3):
        raise ValueError(
            f"Expected `kernel_size` dimension to be 2 or 3. `kernel_size` dimensionality: {len(kernel_size)}"
        )
    if len(sigma) != len(target.shape) - 2:
        raise ValueError(
            f"`kernel_size` has dimension {len(kernel_size)}, but expected to be two less that target dimensionality,"
            f" which is: {len(target.shape)}"
        )
    if len(sigma) not in (2, 3):
        raise ValueError(
            f"Expected `kernel_size` dimension to be 2 or 3. `kernel_size` dimensionality: {len(kernel_size)}"
        )

    if any(x % 2 == 0 or x <= 0 for x in kernel_size):
        raise ValueError(f"Expected `kernel_size` to have odd positive number. Got {kernel_size}.")

    if any(y <= 0 for y in sigma):
        raise ValueError(f"Expected `sigma` to have positive number. Got {sigma}.")

    if data_range is None:
        data_range = max(preds.max() - preds.min(), target.max() - target.min())

    c1 = pow(k1 * data_range, 2)
    c2 = pow(k2 * data_range, 2)
    device = preds.device

    channel = preds.size(1)
    dtype = preds.dtype
    gauss_kernel_size = [int(3.5 * s + 0.5) * 2 + 1 for s in sigma]

    pad_h = (gauss_kernel_size[0] - 1) // 2
    pad_w = (gauss_kernel_size[1] - 1) // 2

    if is_3d:
        pad_d = (gauss_kernel_size[2] - 1) // 2
        preds = _reflection_pad_3d(preds, pad_d, pad_w, pad_h)
        target = _reflection_pad_3d(target, pad_d, pad_w, pad_h)
        if gaussian_kernel:
            kernel = _gaussian_kernel_3d(channel, gauss_kernel_size, sigma, dtype, device)
    else:
        preds = F.pad(preds, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
        target = F.pad(target, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
        if gaussian_kernel:
            kernel = _gaussian_kernel_2d(channel, gauss_kernel_size, sigma, dtype, device)

    if not gaussian_kernel:
        kernel = torch.ones((1, 1, *kernel_size)) / torch.prod(Tensor(kernel_size))

    input_list = torch.cat((preds, target, preds * preds, target * target, preds * target))  # (5 * B, C, H, W)

    if is_3d:
        outputs = F.conv3d(input_list, kernel, groups=channel)
    else:
        outputs = F.conv2d(input_list, kernel, groups=channel)

    output_list = outputs.split(preds.shape[0])

    mu_pred_sq = output_list[0].pow(2)
    mu_target_sq = output_list[1].pow(2)
    mu_pred_target = output_list[0] * output_list[1]

    sigma_pred_sq = output_list[2] - mu_pred_sq
    sigma_target_sq = output_list[3] - mu_target_sq
    sigma_pred_target = output_list[4] - mu_pred_target

    upper = 2 * sigma_pred_target + c2
    lower = sigma_pred_sq + sigma_target_sq + c2

    ssim_idx_full_image = ((2 * mu_pred_target + c1) * upper) / ((mu_pred_sq + mu_target_sq + c1) * lower)

    if is_3d:
        ssim_idx = ssim_idx_full_image[..., pad_h:-pad_h, pad_w:-pad_w, pad_d:-pad_d]
    else:
        ssim_idx = ssim_idx_full_image[..., pad_h:-pad_h, pad_w:-pad_w]

    if return_contrast_sensitivity:
        contrast_sensitivity = upper / lower
        contrast_sensitivity = contrast_sensitivity[..., pad_h:-pad_h, pad_w:-pad_w]
        return reduce(ssim_idx.reshape(ssim_idx.shape[0], -1).mean(-1), reduction), reduce(
            contrast_sensitivity.reshape(contrast_sensitivity.shape[0], -1).mean(-1), reduction
        )

    elif return_full_image:
        return reduce(ssim_idx.reshape(ssim_idx.shape[0], -1).mean(-1), reduction), reduce(
            ssim_idx_full_image, reduction
        )

    return reduce(ssim_idx.reshape(ssim_idx.shape[0], -1).mean(-1), reduction)


def structural_similarity_index_measure(
    preds: Tensor,
    target: Tensor,
    gaussian_kernel: bool = True,
    sigma: Union[float, Sequence[float]] = 1.5,
    kernel_size: Union[int, Sequence[int]] = 11,
    reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
    data_range: Optional[float] = None,
    k1: float = 0.01,
    k2: float = 0.03,
    return_full_image: bool = False,
    return_contrast_sensitivity: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Computes Structual Similarity Index Measure.

    Args:
        preds: estimated image
        target: ground truth image
        gaussian_kernel: If true (default), a gaussian kernel is used, if false a uniform kernel is used
        sigma: Standard deviation of the gaussian kernel, anisotropic kernels are possible.
            Ignored if a uniform kernel is used
        kernel_size: the size of the uniform kernel, anisotropic kernels are possible.
            Ignored if a Gaussian kernel is used
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

        data_range: Range of the image. If ``None``, it is determined from the image (max - min)
        k1: Parameter of SSIM.
        k2: Parameter of SSIM.
        return_full_image: If true, the full ``ssim`` image is returned as a second argument.
            Mutually exclusive with ``return_contrast_sensitivity``
        return_contrast_sensitivity: If true, the constant term is returned as a second argument.
            The luminance term can be obtained with luminance=ssim/contrast
            Mutually exclusive with ``return_full_image``

    Return:
        Tensor with SSIM score

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
        >>> from torchmetrics.functional import structural_similarity_index_measure
        >>> preds = torch.rand([16, 1, 16, 16])
        >>> target = preds * 0.75
        >>> structural_similarity_index_measure(preds, target)
        tensor(0.9219)
    """
    preds, target = _ssim_update(preds, target)
    return _ssim_compute(
        preds,
        target,
        gaussian_kernel,
        sigma,
        kernel_size,
        reduction,
        data_range,
        k1,
        k2,
        return_full_image,
        return_contrast_sensitivity,
    )


def _get_normalized_sim_and_cs(
    preds: Tensor,
    target: Tensor,
    gaussian_kernel: bool = True,
    sigma: Union[float, Sequence[float]] = 1.5,
    kernel_size: Union[int, Sequence[int]] = 11,
    reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
    data_range: Optional[float] = None,
    k1: float = 0.01,
    k2: float = 0.03,
    normalize: Optional[Literal["relu", "simple"]] = None,
) -> Tuple[Tensor, Tensor]:
    sim, contrast_sensitivity = _ssim_compute(
        preds,
        target,
        gaussian_kernel,
        sigma,
        kernel_size,
        reduction,
        data_range,
        k1,
        k2,
        return_contrast_sensitivity=True,
    )
    if normalize == "relu":
        sim = torch.relu(sim)
        contrast_sensitivity = torch.relu(contrast_sensitivity)
    return sim, contrast_sensitivity


def _multiscale_ssim_compute(
    preds: Tensor,
    target: Tensor,
    gaussian_kernel: bool = True,
    sigma: Union[float, Sequence[float]] = 1.5,
    kernel_size: Union[int, Sequence[int]] = 11,
    reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
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
    """Computes Multi-Scale Structual Similarity Index Measure.

    Adapted from: https://github.com/jorge-pessoa/pytorch-msssim/blob/master/pytorch_msssim/__init__.py.

    Args:
        preds: estimated image
        target: ground truth image
        kernel_size: size of the gaussian kernel
        sigma: Standard deviation of the gaussian kernel
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

        data_range: Range of the image. If ``None``, it is determined from the image (max - min)
        k1: Parameter of structural similarity index measure.
        k2: Parameter of structural similarity index measure.
        betas: Exponent parameters for individual similarities and contrastive sensitives returned by different image
            resolutions.
        normalize: When MultiScaleSSIM loss is used for training, it is desirable to use normalizes to improve the
            training stability. This `normalize` argument is out of scope of the original implementation [1], and it is
            adapted from https://github.com/jorge-pessoa/pytorch-msssim instead.

    Raises:
        ValueError:
            If the image height or width is smaller then ``2 ** len(betas)``.
        ValueError:
            If the image height is smaller than ``(kernel_size[0] - 1) * max(1, (len(betas) - 1)) ** 2``.
        ValueError:
            If the image width is smaller than ``(kernel_size[0] - 1) * max(1, (len(betas) - 1)) ** 2``.
    """
    sim_list: List[Tensor] = []
    cs_list: List[Tensor] = []

    is_3d = len(preds.shape) == 5

    if not isinstance(kernel_size, Sequence):
        kernel_size = 3 * [kernel_size] if is_3d else 2 * [kernel_size]
    if not isinstance(sigma, Sequence):
        sigma = 3 * [sigma] if is_3d else 2 * [sigma]

    if preds.size()[-1] < 2 ** len(betas) or preds.size()[-2] < 2 ** len(betas):
        raise ValueError(
            f"For a given number of `betas` parameters {len(betas)}, the image height and width dimensions must be"
            f" larger than or equal to {2 ** len(betas)}."
        )

    _betas_div = max(1, (len(betas) - 1)) ** 2
    if preds.size()[-2] // _betas_div <= kernel_size[0] - 1:
        raise ValueError(
            f"For a given number of `betas` parameters {len(betas)} and kernel size {kernel_size[0]},"
            f" the image height must be larger than {(kernel_size[0] - 1) * _betas_div}."
        )
    if preds.size()[-1] // _betas_div <= kernel_size[1] - 1:
        raise ValueError(
            f"For a given number of `betas` parameters {len(betas)} and kernel size {kernel_size[1]},"
            f" the image width must be larger than {(kernel_size[1] - 1) * _betas_div}."
        )

    for _ in range(len(betas)):
        sim, contrast_sensitivity = _get_normalized_sim_and_cs(
            preds, target, gaussian_kernel, sigma, kernel_size, reduction, data_range, k1, k2, normalize=normalize
        )
        sim_list.append(sim)
        cs_list.append(contrast_sensitivity)
        if len(kernel_size) == 2:
            preds = F.avg_pool2d(preds, (2, 2))
            target = F.avg_pool2d(target, (2, 2))
        elif len(kernel_size) == 3:
            preds = F.avg_pool3d(preds, (2, 2, 2))
            target = F.avg_pool3d(target, (2, 2, 2))
        else:
            raise ValueError("length of kernel_size is neither 2 nor 3")
    sim_stack = torch.stack(sim_list)
    cs_stack = torch.stack(cs_list)

    if normalize == "simple":
        sim_stack = (sim_stack + 1) / 2
        cs_stack = (cs_stack + 1) / 2

    if reduction is None or reduction == "none":
        betas = torch.tensor(betas).unsqueeze(1).repeat(1, sim_stack.shape[0])
        sim_stack = sim_stack ** torch.tensor(betas, device=sim_stack.device)
        cs_stack = cs_stack ** torch.tensor(betas, device=cs_stack.device)
        cs_and_sim = torch.cat((cs_stack[:-1], sim_stack[-1:]), axis=0)
        return torch.prod(cs_and_sim, axis=0)
    else:
        sim_stack = sim_stack ** torch.tensor(betas, device=sim_stack.device)
        cs_stack = cs_stack ** torch.tensor(betas, device=cs_stack.device)
        return torch.prod(cs_stack[:-1]) * sim_stack[-1]


def multiscale_structural_similarity_index_measure(
    preds: Tensor,
    target: Tensor,
    gaussian_kernel: bool = True,
    sigma: Union[float, Sequence[float]] = 1.5,
    kernel_size: Union[int, Sequence[int]] = 11,
    reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
    data_range: Optional[float] = None,
    k1: float = 0.01,
    k2: float = 0.03,
    betas: Tuple[float, ...] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
    normalize: Optional[Literal["relu", "simple"]] = None,
) -> Tensor:
    """Computes `MultiScaleSSIM`_, Multi-scale Structual Similarity Index Measure, which is a generalization of
    Structual Similarity Index Measure by incorporating image details at different resolution scores.

    Args:
        preds: Predictions from model of shape ``[N, C, H, W]``
        target: Ground truth values of shape ``[N, C, H, W]``
        kernel_size: size of the gaussian kernel
        sigma: Standard deviation of the gaussian kernel
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

        data_range: Range of the image. If ``None``, it is determined from the image (max - min)
        k1: Parameter of structural similarity index measure.
        k2: Parameter of structural similarity index measure.
        betas: Exponent parameters for individual similarities and contrastive sensitivies returned by different image
            resolutions.
        normalize: When MultiScaleSSIM loss is used for training, it is desirable to use normalizes to improve the
            training stability. This `normalize` argument is out of scope of the original implementation [1], and it is
            adapted from https://github.com/jorge-pessoa/pytorch-msssim instead.

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
        >>> from torchmetrics.functional import multiscale_structural_similarity_index_measure
        >>> preds = torch.rand([1, 1, 256, 256], generator=torch.manual_seed(42))
        >>> target = preds * 0.75
        >>> multiscale_structural_similarity_index_measure(preds, target)
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
    return _multiscale_ssim_compute(
        preds, target, gaussian_kernel, sigma, kernel_size, reduction, data_range, k1, k2, betas, normalize
    )
