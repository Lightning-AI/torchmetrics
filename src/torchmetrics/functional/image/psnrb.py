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
import math
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, tensor


def _compute_bef(x: Tensor, block_size: int = 8) -> Tensor:
    """Compute block effect.

    Args:
        x: input image
        block_size: integer indication the block size

    Returns:
        Computed block effect

    Raises:
        ValueError:
            If the image is not a grayscale image

    """
    (
        _,
        channels,
        height,
        width,
    ) = x.shape
    if channels > 1:
        raise ValueError(f"`psnrb` metric expects grayscale images, but got images with {channels} channels.")

    h = torch.arange(width - 1)
    h_b = torch.tensor(range(block_size - 1, width - 1, block_size))
    h_bc = torch.tensor(list(set(h.tolist()).symmetric_difference(h_b.tolist())))

    v = torch.arange(height - 1)
    v_b = torch.tensor(range(block_size - 1, height - 1, block_size))
    v_bc = torch.tensor(list(set(v.tolist()).symmetric_difference(v_b.tolist())))

    d_b = (x[:, :, :, h_b] - x[:, :, :, h_b + 1]).pow(2.0).sum()
    d_bc = (x[:, :, :, h_bc] - x[:, :, :, h_bc + 1]).pow(2.0).sum()
    d_b += (x[:, :, v_b, :] - x[:, :, v_b + 1, :]).pow(2.0).sum()
    d_bc += (x[:, :, v_bc, :] - x[:, :, v_bc + 1, :]).pow(2.0).sum()

    n_hb = height * (width / block_size) - 1
    n_hbc = (height * (width - 1)) - n_hb
    n_vb = width * (height / block_size) - 1
    n_vbc = (width * (height - 1)) - n_vb
    d_b /= n_hb + n_vb
    d_bc /= n_hbc + n_vbc
    t = math.log2(block_size) / math.log2(min(height, width)) if d_b > d_bc else 0
    return t * (d_b - d_bc)


def _psnrb_compute(
    sum_squared_error: Tensor,
    bef: Tensor,
    n_obs: Tensor,
    data_range: Tensor,
) -> Tensor:
    """Computes peak signal-to-noise ratio.

    Args:
        sum_squared_error: Sum of square of errors over all observations
        bef: block effect
        n_obs: Number of predictions or observations
        data_range: the range of the data. If None, it is determined from the data (max - min).
    """
    sum_squared_error = sum_squared_error / n_obs + bef
    if data_range > 2:
        return 10 * torch.log10(data_range**2 / sum_squared_error)
    return 10 * torch.log10(1.0 / sum_squared_error)


def _psnrb_update(preds: Tensor, target: Tensor, block_size: int = 8) -> Tuple[Tensor, Tensor, Tensor]:
    """Updates and returns variables required to compute peak signal-to-noise ratio.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        block_size: Integer indication the block size
    """
    sum_squared_error = torch.sum(torch.pow(preds - target, 2))
    n_obs = tensor(target.numel(), device=target.device)
    bef = _compute_bef(preds, block_size=block_size)
    return sum_squared_error, bef, n_obs


def peak_signal_noise_ratio_with_blocked_effect(
    preds: Tensor,
    target: Tensor,
    block_size: int = 8,
) -> Tensor:
    r"""Computes `Peak Signal to Noise Ratio With Blocked Effect` (PSNRB) metrics.

    .. math::
        \text{PSNRB}(I, J) = 10 * \log_{10} \left(\frac{\max(I)^2}{\text{MSE}(I, J)-\text{B}(I, J)}\right)

    Where :math:`\text{MSE}` denotes the `mean-squared-error`_ function.

    Args:
        preds: estimated signal
        target: groun truth signal
        block_size: integer indication the block size

    Return:
        Tensor with PSNRB score

    Example:
        >>> import torch
        >>> from torchmetrics.functional.image import peak_signal_noise_ratio_with_blocked_effect
        >>> _ = torch.manual_seed(42)
        >>> preds = torch.rand(1, 1, 28, 28)
        >>> target = torch.rand(1, 1, 28, 28)
        >>> peak_signal_noise_ratio_with_blocked_effect(preds, target)
        tensor(7.8402)
    """
    data_range = target.max() - target.min()
    sum_squared_error, bef, n_obs = _psnrb_update(preds, target, block_size=block_size)
    return _psnrb_compute(sum_squared_error, bef, n_obs, data_range)
