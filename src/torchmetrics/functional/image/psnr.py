# Copyright The Lightning team.
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
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, tensor
from typing_extensions import Literal

from torchmetrics.utilities import rank_zero_warn, reduce


def _psnr_compute(
    sum_squared_error: Tensor,
    n_obs: Tensor,
    data_range: Tensor,
    base: float = 10.0,
    reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
) -> Tensor:
    """Compute peak signal-to-noise ratio.

    Args:
        sum_squared_error: Sum of square of errors over all observations
        n_obs: Number of predictions or observations
        data_range: the range of the data. If None, it is determined from the data (max - min).
           ``data_range`` must be given when ``dim`` is not None.
        base: a base of a logarithm to use
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

    Example:
        >>> preds = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        >>> target = torch.tensor([[3.0, 2.0], [1.0, 0.0]])
        >>> data_range = target.max() - target.min()
        >>> sum_squared_error, n_obs = _psnr_update(preds, target)
        >>> _psnr_compute(sum_squared_error, n_obs, data_range)
        tensor(2.5527)
    """
    psnr_base_e = 2 * torch.log(data_range) - torch.log(sum_squared_error / n_obs)
    psnr_vals = psnr_base_e * (10 / torch.log(tensor(base)))
    return reduce(psnr_vals, reduction=reduction)


def _psnr_update(
    preds: Tensor,
    target: Tensor,
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
) -> Tuple[Tensor, Tensor]:
    """Update and return variables required to compute peak signal-to-noise ratio.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        dim: Dimensions to reduce PSNR scores over provided as either an integer or a list of integers.
            Default is None meaning scores will be reduced across all dimensions.
    """
    if dim is None:
        sum_squared_error = torch.sum(torch.pow(preds - target, 2))
        n_obs = tensor(target.numel(), device=target.device)
        return sum_squared_error, n_obs

    diff = preds - target
    sum_squared_error = torch.sum(diff * diff, dim=dim)

    dim_list = [dim] if isinstance(dim, int) else list(dim)
    if not dim_list:
        n_obs = tensor(target.numel(), device=target.device)
    else:
        n_obs = tensor(target.size(), device=target.device)[dim_list].prod()
        n_obs = n_obs.expand_as(sum_squared_error)

    return sum_squared_error, n_obs


def peak_signal_noise_ratio(
    preds: Tensor,
    target: Tensor,
    data_range: Optional[Union[float, Tuple[float, float]]] = None,
    base: float = 10.0,
    reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
) -> Tensor:
    """Compute the peak signal-to-noise ratio.

    Args:
        preds: estimated signal
        target: groun truth signal
        data_range:
            the range of the data. If None, it is determined from the data (max - min). If a tuple is provided then
            the range is calculated as the difference and input is clamped between the values.
            The ``data_range`` must be given when ``dim`` is not None.
        base: a base of a logarithm to use
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'`` or None``: no reduction will be applied

        dim:
            Dimensions to reduce PSNR scores over provided as either an integer or a list of integers. Default is
            None meaning scores will be reduced across all dimensions.

    Return:
        Tensor with PSNR score

    Raises:
        ValueError:
            If ``dim`` is not ``None`` and ``data_range`` is not provided.

    Example:
        >>> from torchmetrics.functional.image import peak_signal_noise_ratio
        >>> pred = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        >>> target = torch.tensor([[3.0, 2.0], [1.0, 0.0]])
        >>> peak_signal_noise_ratio(pred, target)
        tensor(2.5527)

    .. note::
        Half precision is only support on GPU for this metric
    """
    if dim is None and reduction != "elementwise_mean":
        rank_zero_warn(f"The `reduction={reduction}` will not have any effect when `dim` is None.")

    if data_range is None:
        if dim is not None:
            # Maybe we could use `torch.amax(target, dim=dim) - torch.amin(target, dim=dim)` in PyTorch 1.7 to calculate
            # `data_range` in the future.
            raise ValueError("The `data_range` must be given when `dim` is not None.")

        data_range = target.max() - target.min()
    elif isinstance(data_range, tuple):
        preds = torch.clamp(preds, min=data_range[0], max=data_range[1])
        target = torch.clamp(target, min=data_range[0], max=data_range[1])
        data_range = tensor(data_range[1] - data_range[0])
    else:
        data_range = tensor(float(data_range))

    sum_squared_error, n_obs = _psnr_update(preds, target, dim=dim)
    return _psnr_compute(sum_squared_error, n_obs, data_range, base=base, reduction=reduction)
