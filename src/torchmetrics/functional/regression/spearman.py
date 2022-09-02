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

from torchmetrics.utilities.checks import _check_same_shape


def _find_repeats(data: Tensor) -> Tensor:
    """find and return values which have repeats i.e. the same value are more than once in the tensor."""
    temp = data.detach().clone()
    temp = temp.sort()[0]

    change = torch.cat([torch.tensor([True], device=temp.device), temp[1:] != temp[:-1]])
    unique = temp[change]
    change_idx = torch.cat([torch.nonzero(change), torch.tensor([[temp.numel()]], device=temp.device)]).flatten()
    freq = change_idx[1:] - change_idx[:-1]
    atleast2 = freq > 1
    return unique[atleast2]


def _rank_data(data: Tensor) -> Tensor:
    """Calculate the rank for each element of a tensor.

    The rank refers to the indices of an element in the corresponding sorted tensor (starting from 1).
    Duplicates of the same value will be assigned the mean of their rank.

    Adopted from `Rank of element tensor`_
    """
    n = data.numel()
    rank = torch.empty_like(data)
    idx = data.argsort()
    rank[idx[:n]] = torch.arange(1, n + 1, dtype=data.dtype, device=data.device)

    repeats = _find_repeats(data)
    for r in repeats:
        condition = data == r
        rank[condition] = rank[condition].mean()
    return rank


def _spearman_corrcoef_update(preds: Tensor, target: Tensor, n_out: int) -> Tuple[Tensor, Tensor]:
    """Updates and returns variables required to compute Spearman Correlation Coefficient.

    Checks for same shape and type of input tensors.

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
    if preds.ndim > 2 or target.ndim > 2:
        raise ValueError(
            "Expected both predictions and target to be either 1 or 2 dimensional tensors,"
            " but get{target.ndim} and {preds.ndim}."
        )
    if (n_out == 1 and preds.ndim != 1) or (n_out > 1 and n_out != preds.shape[-1]):
        raise ValueError(
            "Expected argument `num_outputs` to match the second dimension of input, but got {self.n_out}"
            " and {preds.ndim}."
        )

    return preds, target


def _spearman_corrcoef_compute(preds: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    """Computes Spearman Correlation Coefficient.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        eps: Avoids ``ZeroDivisionError``.

    Example:
        >>> target = torch.tensor([3, -0.5, 2, 7])
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> preds, target = _spearman_corrcoef_update(preds, target, n_out=1)
        >>> _spearman_corrcoef_compute(preds, target)
        tensor(1.0000)
    """
    if preds.ndim == 1:
        preds = _rank_data(preds)
        target = _rank_data(target)
    else:
        preds = torch.stack([_rank_data(p) for p in preds.T]).T
        target = torch.stack([_rank_data(t) for t in target.T]).T

    preds_diff = preds - preds.mean(0)
    target_diff = target - target.mean(0)

    cov = (preds_diff * target_diff).mean(0)
    preds_std = torch.sqrt((preds_diff * preds_diff).mean(0))
    target_std = torch.sqrt((target_diff * target_diff).mean(0))

    corrcoef = cov / (preds_std * target_std + eps)
    return torch.clamp(corrcoef, -1.0, 1.0)


def spearman_corrcoef(preds: Tensor, target: Tensor) -> Tensor:
    r"""
     Computes `spearmans rank correlation coefficient`_:

    .. math:
        r_s = = \frac{cov(rg_x, rg_y)}{\sigma_{rg_x} * \sigma_{rg_y}}

    where :math:`rg_x` and :math:`rg_y` are the rank associated to the variables x and y. Spearmans correlations
    coefficient corresponds to the standard pearsons correlation coefficient calculated on the rank variables.

    Args:
        preds: estimated scores
        target: ground truth scores

    Example:
        >>> from torchmetrics.functional import spearman_corrcoef
        >>> target = torch.tensor([3, -0.5, 2, 7])
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> spearman_corrcoef(preds, target)
        tensor(1.0000)

    """
    preds, target = _spearman_corrcoef_update(preds, target, n_out=1 if preds.ndim == 1 else preds.shape[-1])
    return _spearman_corrcoef_compute(preds, target)
