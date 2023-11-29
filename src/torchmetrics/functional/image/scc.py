from typing import Tuple, Union

import torch
from torch import Tensor, tensor
from torch.nn.functional import conv2d

from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.distributed import reduce


def _scc_update(preds: Tensor, target: Tensor, hp_filter: Tensor, window_size: int) -> Tuple[Tensor, Tensor, Tensor]:
    """Update and returns variables required to compute Spatial Correlation Coefficient.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        hp_filter: High-pass filter tensor
        window_size: Local window size integer

    Return:
        Tuple of (preds, target, hp_filter) tensors

    Raises:
        ValueError:
            If ``preds`` and ``target`` have different number of channels
            If ``preds`` and ``target`` have different shapes
            If ``preds`` and ``target`` have invalid shapes
            If ``window_size`` is not a positive integer
            If ``window_size`` is greater than the size of the image

    """
    if preds.dtype != target.dtype:
        target = target.to(preds.dtype)
    _check_same_shape(preds, target)
    if len(preds.shape) not in (3, 4):
        raise ValueError(
            "Expected `preds` and `target` to have batch of colored images with BxCxHxW shape"
            "  or batch of grayscale images of BxHxW shape."
            f" Got preds: {preds.shape} and target: {target.shape}."
        )

    if len(preds.shape) == 3:
        preds = preds.unsqueeze(1)
        target = target.unsqueeze(1)

    if not window_size > 0:
        raise ValueError(f"Expected `window_size` to be a positive integer. Got {window_size}.")

    if window_size > preds.size(2) or window_size > preds.size(3):
        raise ValueError(
            f"Expected `window_size` to be less than or equal to the size of the image."
            f" Got window_size: {window_size} and image size: {preds.size(2)}x{preds.size(3)}."
        )

    preds = preds.to(torch.float32)
    target = target.to(torch.float32)
    hp_filter = hp_filter[None, None, :].to(dtype=preds.dtype, device=preds.device)
    return preds, target, hp_filter


def _symmetric_reflect_pad_2d(input: Tensor, pad: Union[int, Tuple[int, ...]]) -> Tensor:
    if isinstance(pad, int):
        pad = (pad, pad, pad, pad)
    assert len(pad) == 4

    left_pad = input[:, :, :, 0 : pad[0]].flip(dims=[3])
    right_pad = input[:, :, :, -pad[1] :].flip(dims=[3])
    padded = torch.cat([left_pad, input, right_pad], dim=3)

    top_pad = padded[:, :, 0 : pad[2], :].flip(dims=[2])
    bottom_pad = padded[:, :, -pad[3] :, :].flip(dims=[2])
    return torch.cat([top_pad, padded, bottom_pad], dim=2)


def _signal_convolve_2d(input: Tensor, kernel: Tensor) -> Tensor:
    left_padding = int(torch.floor(tensor((kernel.size(3) - 1) / 2)).item())
    right_padding = int(torch.ceil(tensor((kernel.size(3) - 1) / 2)).item())
    top_padding = int(torch.floor(tensor((kernel.size(2) - 1) / 2)).item())
    bottom_padding = int(torch.ceil(tensor((kernel.size(2) - 1) / 2)).item())

    padded = _symmetric_reflect_pad_2d(input, pad=(left_padding, right_padding, top_padding, bottom_padding))
    kernel = kernel.flip([2, 3])
    return conv2d(padded, kernel, stride=1, padding=0)


def _hp_2d_laplacian(input: Tensor, kernel: Tensor) -> Tensor:
    output = _signal_convolve_2d(input, kernel)
    output += _signal_convolve_2d(input, kernel)
    return output


def _local_variance_covariance(preds: Tensor, target: Tensor, window: Tensor):
    preds_mean = conv2d(preds, window, stride=1, padding="same")
    target_mean = conv2d(target, window, stride=1, padding="same")

    preds_var = conv2d(preds**2, window, stride=1, padding="same") - preds_mean**2
    target_var = conv2d(target**2, window, stride=1, padding="same") - target_mean**2
    target_preds_cov = conv2d(target * preds, window, stride=1, padding="same") - target_mean * preds_mean

    return preds_var, target_var, target_preds_cov


def _scc_per_channel_compute(preds: Tensor, target: Tensor, hp_filter: Tensor, window_size: int):
    """Computes per channel Spatial Correlation Coefficient.

    Args:
        preds: estimated image of Bx1xHxW shape.
        target: ground truth image of Bx1xHxW shape.
        hp_filter: 2D high-pass filter.
        window_size: size of window for local mean calculation.

    Return:
        Tensor with Spatial Correlation Coefficient score

    """
    dtype = preds.dtype
    device = preds.device

    # This code is inspired by
    # https://github.com/andrewekhalel/sewar/blob/master/sewar/full_ref.py#L187.

    window = torch.ones(size=(1, 1, window_size, window_size), dtype=dtype, device=device) / (window_size**2)

    preds_hp = _hp_2d_laplacian(preds, hp_filter)
    target_hp = _hp_2d_laplacian(target, hp_filter)

    preds_var, target_var, target_preds_cov = _local_variance_covariance(preds_hp, target_hp, window)

    preds_var[preds_var < 0] = 0
    target_var[target_var < 0] = 0

    den = torch.sqrt(target_var) * torch.sqrt(preds_var)
    idx = den == 0
    den[den == 0] = 1
    scc = target_preds_cov / den
    scc[idx] = 0
    return scc


def spatial_correlation_coefficient(
    preds: Tensor,
    target: Tensor,
    hp_filter: Tensor = tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
    window_size: int = 8,
):
    """Compute Spatial Correlation Coefficient (SCC_).

    Args:
        preds: predicted images of shape ``(N,C,H,W)`` or ``(N,H,W)``.
        target: ground truth images of shape ``(N,C,H,W)`` or ``(N,H,W)``.
        hp_filter: High-pass filter tensor. default: tensor([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
        window_size: Local window size integer. default: 8

    Return:
        Tensor with scc score

    Example:
        >>> import torch
        >>> from torchmetrics.functional.image import spatial_correlation_coefficient as scc
        >>> _ = torch.manual_seed(42)
        >>> x = torch.randn(5, 3, 16, 16)
        >>> scc(x, x)
        tensor(1.)
        >>> x = torch.randn(5, 16, 16)
        >>> scc(x, x)
        tensor(1.)

    """
    preds, target, hp_filter = _scc_update(preds, target, hp_filter, window_size)

    per_channel = [
        _scc_per_channel_compute(
            preds[:, i, :, :].unsqueeze(1), target[:, i, :, :].unsqueeze(1), hp_filter, window_size
        )
        for i in range(preds.size(1))
    ]
    return reduce(torch.cat(per_channel, dim=1), reduction="elementwise_mean")
