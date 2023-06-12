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
from typing import Optional, Tuple

import torch
from torch import Tensor

from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_1_9


def _safe_matmul(x: Tensor, y: Tensor) -> Tensor:
    """Safe calculation of matrix multiplication.

    If input is float16, will cast to float32 for computation and back again.
    """
    if x.dtype == torch.float16 or y.dtype == torch.float16:
        return (x.float() @ y.T.float()).half()
    return x @ y.T


def _safe_xlogy(x: Tensor, y: Tensor) -> Tensor:
    """Compute x * log(y). Returns 0 if x=0.

    Example:
        >>> import torch
        >>> x = torch.zeros(1)
        >>> _safe_xlogy(x, 1/x)
        tensor([0.])

    """
    res = x * torch.log(y)
    res[x == 0] = 0.0
    return res


def _safe_divide(num: Tensor, denom: Tensor) -> Tensor:
    """Safe division, by preventing division by zero.

    Additionally casts to float if input is not already to secure backwards compatibility.
    """
    denom[denom == 0.0] = 1
    num = num if num.is_floating_point() else num.float()
    denom = denom if denom.is_floating_point() else denom.float()
    return num / denom


def _adjust_weights_safe_divide(
    score: Tensor, average: Optional[str], multilabel: bool, tp: Tensor, fp: Tensor, fn: Tensor
) -> Tensor:
    if average is None or average == "none":
        return score
    if average == "weighted":
        weights = tp + fn
    else:
        weights = torch.ones_like(score)
        if not multilabel:
            weights[tp + fp + fn == 0] = 0.0
    return _safe_divide(weights * score, weights.sum(-1, keepdim=True)).sum(-1)


def _auc_format_inputs(x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
    """Check that auc input is correct."""
    x = x.squeeze() if x.ndim > 1 else x
    y = y.squeeze() if y.ndim > 1 else y

    if x.ndim > 1 or y.ndim > 1:
        raise ValueError(
            f"Expected both `x` and `y` tensor to be 1d, but got tensors with dimension {x.ndim} and {y.ndim}"
        )
    if x.numel() != y.numel():
        raise ValueError(
            f"Expected the same number of elements in `x` and `y` tensor but received {x.numel()} and {y.numel()}"
        )
    return x, y


def _auc_compute_without_check(x: Tensor, y: Tensor, direction: float, axis: int = -1) -> Tensor:
    """Compute area under the curve using the trapezoidal rule.

    Assumes increasing or decreasing order of `x`.
    """
    with torch.no_grad():
        auc_: Tensor = torch.trapz(y, x, dim=axis) * direction
    return auc_


def _auc_compute(x: Tensor, y: Tensor, reorder: bool = False) -> Tensor:
    with torch.no_grad():
        if reorder:
            x, x_idx = torch.sort(x, stable=True) if _TORCH_GREATER_EQUAL_1_9 else torch.sort(x)
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
    """Compute Area Under the Curve (AUC) using the trapezoidal rule.

    Args:
        x: x-coordinates, must be either increasing or decreasing
        y: y-coordinates
        reorder: if True, will reorder the arrays to make it either increasing or decreasing

    Return:
        Tensor containing AUC score
    """
    x, y = _auc_format_inputs(x, y)
    return _auc_compute(x, y, reorder=reorder)
