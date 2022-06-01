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


def _auc_update(x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
    """Updates and returns variables required to compute area under the curve. Checks if the 2 input tenser have
    the same number of elements and if they are 1d.

    Args:
        x: x-coordinates
        y: y-coordinates
    """

    if x.ndim > 1:
        x = x.squeeze()

    if y.ndim > 1:
        y = y.squeeze()

    if x.ndim > 1 or y.ndim > 1:
        raise ValueError(
            f"Expected both `x` and `y` tensor to be 1d, but got tensors with dimension {x.ndim} and {y.ndim}"
        )
    if x.numel() != y.numel():
        raise ValueError(
            f"Expected the same number of elements in `x` and `y` tensor but received {x.numel()} and {y.numel()}"
        )
    return x, y


def _auc_compute_without_check(x: Tensor, y: Tensor, direction: float) -> Tensor:
    """Computes area under the curve using the trapezoidal rule. Assumes increasing or decreasing order of `x`.

    Args:
        x: x-coordinates, must be either increasing or decreasing
        y: y-coordinates
        direction: 1 if increaing, -1 if decreasing

    Example:
        >>> x = torch.tensor([0, 1, 2, 3])
        >>> y = torch.tensor([0, 1, 2, 2])
        >>> x, y = _auc_update(x, y)
        >>> _auc_compute_without_check(x, y, direction=1.0)
        tensor(4.)
    """

    with torch.no_grad():
        auc_: Tensor = torch.trapz(y, x) * direction
    return auc_


def _auc_compute(x: Tensor, y: Tensor, reorder: bool = False) -> Tensor:
    """Computes area under the curve using the trapezoidal rule. Checks for increasing or decreasing order of `x`.

    Args:
        x: x-coordinates, must be either increasing or decreasing
        y: y-coordinates
        reorder: if True, will reorder the arrays to make it either increasing or decreasing

    Example:
        >>> x = torch.tensor([0, 1, 2, 3])
        >>> y = torch.tensor([0, 1, 2, 2])
        >>> x, y = _auc_update(x, y)
        >>> _auc_compute(x, y)
        tensor(4.)
        >>> _auc_compute(x, y, reorder=True)
        tensor(4.)
    """

    with torch.no_grad():
        if reorder:
            # TODO: include stable=True arg when pytorch v1.9 is released
            x, x_idx = torch.sort(x)
            y = y[x_idx]

        dx = x[1:] - x[:-1]
        if (dx < 0).any():
            if (dx <= 0).all():
                direction = -1.0
            else:
                raise ValueError(
                    "The `x` tensor is neither increasing or decreasing. Try setting the reorder argument to `True`."
                )
        else:
            direction = 1.0
        return _auc_compute_without_check(x, y, direction)


def auc(x: Tensor, y: Tensor, reorder: bool = False) -> Tensor:
    """Computes Area Under the Curve (AUC) using the trapezoidal rule.

    Args:
        x: x-coordinates, must be either increasing or decreasing
        y: y-coordinates
        reorder: if True, will reorder the arrays to make it either increasing or decreasing

    Return:
        Tensor containing AUC score

    Raises:
        ValueError:
            If both ``x`` and ``y`` tensors are not ``1d``.
        ValueError:
            If both ``x`` and ``y`` don't have the same numnber of elements.
        ValueError:
            If ``x`` tesnsor is neither increasing nor decreasing.

    Example:
        >>> from torchmetrics.functional import auc
        >>> x = torch.tensor([0, 1, 2, 3])
        >>> y = torch.tensor([0, 1, 2, 2])
        >>> auc(x, y)
        tensor(4.)
        >>> auc(x, y, reorder=True)
        tensor(4.)
    """
    x, y = _auc_update(x, y)
    return _auc_compute(x, y, reorder=reorder)
