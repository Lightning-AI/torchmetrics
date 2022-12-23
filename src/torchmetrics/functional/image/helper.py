from typing import Sequence, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_1_10


def _gaussian(kernel_size: int, sigma: float, dtype: torch.dtype, device: torch.device) -> Tensor:
    """Computes 1D gaussian kernel.

    Args:
        kernel_size: size of the gaussian kernel
        sigma: Standard deviation of the gaussian kernel
        dtype: data type of the output tensor
        device: device of the output tensor

    Example:
        >>> _gaussian(3, 1, torch.float, 'cpu')
        tensor([[0.2741, 0.4519, 0.2741]])
    """
    dist = torch.arange(start=(1 - kernel_size) / 2, end=(1 + kernel_size) / 2, step=1, dtype=dtype, device=device)
    gauss = torch.exp(-torch.pow(dist / sigma, 2) / 2)
    return (gauss / gauss.sum()).unsqueeze(dim=0)  # (1, kernel_size)


def _gaussian_kernel_2d(
    channel: int,
    kernel_size: Sequence[int],
    sigma: Sequence[float],
    dtype: torch.dtype,
    device: Union[torch.device, str],
) -> Tensor:
    """Computes 2D gaussian kernel.

    Args:
        channel: number of channels in the image
        kernel_size: size of the gaussian kernel as a tuple (h, w)
        sigma: Standard deviation of the gaussian kernel
        dtype: data type of the output tensor
        device: device of the output tensor

    Example:
        >>> _gaussian_kernel_2d(1, (5,5), (1,1), torch.float, "cpu")
        tensor([[[[0.0030, 0.0133, 0.0219, 0.0133, 0.0030],
                  [0.0133, 0.0596, 0.0983, 0.0596, 0.0133],
                  [0.0219, 0.0983, 0.1621, 0.0983, 0.0219],
                  [0.0133, 0.0596, 0.0983, 0.0596, 0.0133],
                  [0.0030, 0.0133, 0.0219, 0.0133, 0.0030]]]])
    """

    gaussian_kernel_x = _gaussian(kernel_size[0], sigma[0], dtype, device)
    gaussian_kernel_y = _gaussian(kernel_size[1], sigma[1], dtype, device)
    kernel = torch.matmul(gaussian_kernel_x.t(), gaussian_kernel_y)  # (kernel_size, 1) * (1, kernel_size)

    return kernel.expand(channel, 1, kernel_size[0], kernel_size[1])


def _gaussian_kernel_3d(
    channel: int, kernel_size: Sequence[int], sigma: Sequence[float], dtype: torch.dtype, device: torch.device
) -> Tensor:
    """Computes 3D gaussian kernel.

    Args:
        channel: number of channels in the image
        kernel_size: size of the gaussian kernel as a tuple (h, w, d)
        sigma: Standard deviation of the gaussian kernel
        dtype: data type of the output tensor
        device: device of the output tensor
    """

    gaussian_kernel_x = _gaussian(kernel_size[0], sigma[0], dtype, device)
    gaussian_kernel_y = _gaussian(kernel_size[1], sigma[1], dtype, device)
    gaussian_kernel_z = _gaussian(kernel_size[2], sigma[2], dtype, device)
    kernel_xy = torch.matmul(gaussian_kernel_x.t(), gaussian_kernel_y)  # (kernel_size, 1) * (1, kernel_size)
    kernel = torch.mul(
        kernel_xy.unsqueeze(-1).repeat(1, 1, kernel_size[2]),
        gaussian_kernel_z.expand(kernel_size[0], kernel_size[1], kernel_size[2]),
    )
    return kernel.expand(channel, 1, kernel_size[0], kernel_size[1], kernel_size[2])


def _single_dimension_pad(inputs: Tensor, dim: int, pad: int) -> Tensor:
    """Reflective padding of input along a specific dimension.

    Args:
        inputs: tensor to pad
        dim: dimension to pad along
        pad: amount of padding to add

    Returns:
        padded input
    """
    _max = inputs.shape[dim] - 2
    x = torch.index_select(inputs, dim, torch.arange(pad, 0, -1, device=inputs.device))
    y = torch.index_select(inputs, dim, torch.arange(_max, _max - pad, -1, device=inputs.device))
    return torch.cat((x, inputs, y), dim)


def _reflection_pad_3d(inputs: Tensor, pad_h: int, pad_w: int, pad_d: int) -> Tensor:
    """Reflective padding of 3d input.

    Args:
        inputs: tensor to pad, should be a 3D tensor of shape ``[N, C, H, W, D]``
        pad_w: amount of padding in the height dimension
        pad_h: amount of padding in the width dimension
        pad_d: amount of padding in the depth dimension

    Returns:
        padded input tensor
    """
    if _TORCH_GREATER_EQUAL_1_10:
        inputs = F.pad(inputs, (pad_h, pad_h, pad_w, pad_w, pad_d, pad_d), mode="reflect")
    else:
        rank_zero_warn(
            "An older version of pyTorch is used. For optimal speed, please upgrade to at least pyTorch 1.10."
        )
        for dim, pad in enumerate([pad_h, pad_w, pad_d]):
            inputs = _single_dimension_pad(inputs, dim + 2, pad)
    return inputs
