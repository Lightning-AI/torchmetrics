from typing import Sequence

import torch
from torch import Tensor


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


def _gaussian_kernel(
    channel: int, kernel_size: Sequence[int], sigma: Sequence[float], dtype: torch.dtype, device: torch.device
) -> Tensor:
    """Computes 2D gaussian kernel.

    Args:
        channel: number of channels in the image
        kernel_size: size of the gaussian kernel as a tuple (h, w)
        sigma: Standard deviation of the gaussian kernel
        dtype: data type of the output tensor
        device: device of the output tensor

    Example:
        >>> _gaussian_kernel(1, (5,5), (1,1), torch.float, "cpu")
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
