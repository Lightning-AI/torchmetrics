from typing import Sequence, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F


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


def _get_uniform_weight_and_bias_for_conv2d(inputs: Tensor, window_size: int) -> Tuple[Tensor, Tensor]:
    """Construct uniform weight and bias for a 2d convolution.

    Args:
        inputs: Input image
        window_size: size of convolutional kernel

    Return:
        The weight and bias for 2d convolution
    """
    _channels = inputs.shape[1]
    kernel_weight = torch.ones(1, _channels, window_size, window_size, dtype=inputs.dtype, device=inputs.device)
    kernel_weight /= _channels * window_size * window_size
    kernel_bias = torch.tensor([0.0], dtype=inputs.dtype, device=inputs.device)
    return kernel_weight, kernel_bias


def _single_dimension_pad(inputs: Tensor, dim: int, pad: int) -> Tensor:
    """Applies single-dimension reflection padding to match scipy implementation:

    Args:
        inputs: Input image
        dim: A dimension the image should be padded over
        pad: Number of pads

    Return:
        Image padded over a single dimension
    """
    _max = inputs.shape[dim] - 2
    x = torch.index_select(inputs, dim, torch.arange(pad - 1, -1, -1))
    y = torch.index_select(inputs, dim, torch.arange(_max - 1, _max - pad, -1))
    return torch.cat((x, inputs, y), dim)


def _reflection_pad2d(inputs: Tensor, pad: int) -> Tensor:
    """Applies reflection padding to the input image.

    Args:
        inputs: Input image
        dim: A dimension the image should be padded over
        pad: Number of pads

    Return:
        Padded image
    """
    for dim in [2, 3]:
        inputs = _single_dimension_pad(inputs, dim, pad)
    return inputs


def _uniform_filter(inputs: Tensor, window_size: int) -> Tensor:
    """Applies uniform filtew with a window of a given size over the input image.

    Args:
        inputs: Input image
        window_size: Sliding window used for rmse calculation
        kernel_weight:


    Return:
        Image transformed with the uniform input
    """
    inputs = _reflection_pad2d(inputs, window_size // 2)
    kernel_weight, kernel_bias = _get_uniform_weight_and_bias_for_conv2d(inputs, window_size)
    inputs = F.conv2d(inputs, kernel_weight, kernel_bias)
    return inputs
