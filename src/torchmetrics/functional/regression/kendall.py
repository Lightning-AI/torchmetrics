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

from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.data import _bincount


def _sort_on_first_sequence(x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
    x, perm = x.sort(stable=False)
    for i in range(x.shape[0]):
        y[i] = y[i][perm[i]]
    return x, y


def _count_concordant_pairs(preds: Tensor, target: Tensor) -> Tensor:
    """Count a total number of concordant pairs in given sequences."""

    def _concordant_element_sum(x: Tensor, y: Tensor, i: int) -> Tensor:
        return torch.logical_and(x[i] < x[(i + 1) :], y[i] < y[(i + 1) :]).sum(0).unsqueeze(0)

    return torch.cat([_concordant_element_sum(preds, target, i) for i in range(preds.shape[0])]).sum(0)


def _count_discordant_pairs(preds: Tensor, target: Tensor) -> Tensor:
    """Count a total number of discordant pairs in given sequences."""

    def _discordant_element_sum(x: Tensor, y: Tensor, i: int) -> Tensor:
        return (
            torch.logical_or(
                torch.logical_and(x[i] > x[(i + 1) :], y[i] < y[(i + 1) :]),
                torch.logical_and(x[i] < x[(i + 1) :], y[i] > y[(i + 1) :]),
            )
            .sum(0)
            .unsqueeze(0)
        )

    return torch.cat([_discordant_element_sum(preds, target, i) for i in range(preds.shape[0])]).sum(0)


def _convert_sequence_to_dense_rank(x: Tensor) -> Tensor:
    _ones = torch.zeros(1, x.shape[1], dtype=torch.int32, device=x.device)
    return torch.cat([_ones, (x[1:] != x[:-1]).int()], dim=0).cumsum(0)


def _get_ties(x: Tensor) -> Tensor:
    ties = torch.zeros(x.shape[1], dtype=x.dtype, device=x.device)
    for dim in range(x.shape[1]):
        n_ties = _bincount(x[:, dim])
        n_ties = n_ties[n_ties > 1]
        ties[dim] = (n_ties * (n_ties - 1) // 2).sum()
    return ties


def _kendall_corrcoef_update(
    preds: Tensor,
    target: Tensor,
    concordant_pairs: Tensor,
    discordant_pairs: Tensor,
    preds_ties: Optional[Tensor],
    target_ties: Optional[Tensor],
    total: Optional[Tensor],
    num_outputs: int,
    variant: Literal["a", "b", "c"] = "b",
) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    """Update variables required to compute Kendall rank correlation coefficient.

    Check for the same shape of input tensors

    Args:
        preds: Ordered sequence of data
        target: Ordered sequence of data

    Raises:
        RuntimeError: If ``preds`` and ``target`` do not have the same shape
    """
    # Data checking
    _check_same_shape(preds, target)
    if preds.ndim > 2 or target.ndim > 2:
        raise ValueError(
            f"Expected both predictions and target to be either 1- or 2-dimensional tensors,"
            f" but got {target.ndim} and {preds.ndim}."
        )
    if (num_outputs == 1 and preds.ndim != 1) or (num_outputs > 1 and num_outputs != preds.shape[-1]):
        raise ValueError(
            f"Expected argument `num_outputs` to match the second dimension of input, but got {num_outputs}"
            f" and {preds.ndim}."
        )
    if num_outputs == 1:
        preds = preds.unsqueeze(0)
        target = target.unsqueeze(0)

    # Sort on target and convert it to dense rank
    preds, target = _sort_on_first_sequence(preds, target)
    preds, target = preds.T, target.T  # [num_outputs, seq_len]

    concordant_pairs += _count_concordant_pairs(preds, target)
    discordant_pairs += _count_discordant_pairs(preds, target)

    if variant == "b":
        preds = _convert_sequence_to_dense_rank(preds)
        target = _convert_sequence_to_dense_rank(target)
        preds_ties += _get_ties(preds)
        target_ties += _get_ties(target)
        total += preds.shape[0]

    return concordant_pairs, discordant_pairs, preds_ties, target_ties


def _kendall_corrcoef_compute(
    concordant_pairs: Tensor,
    discordant_pairs: Tensor,
    preds_ties: Optional[Tensor],
    target_ties: Optional[Tensor],
    total: Optional[Tensor],
    variant: Literal["a", "b", "c"] = "a",
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    con_min_dis_pairs = concordant_pairs - discordant_pairs

    if variant == "a":
        return con_min_dis_pairs / (concordant_pairs + discordant_pairs)
    if variant == "b":
        combinations = total * (total - 1) // 2
        denominator = (combinations - preds_ties) * (combinations - target_ties)
        return con_min_dis_pairs / torch.sqrt(denominator)


def kendall_rank_corrcoef(
    preds: Tensor, target: Tensor, variant: Literal["a", "b", "c"] = "b"
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Compute Kendall rank correlation coefficient, commonly also known as Kendall's tau.

    Args:
        preds: Ordered sequence of data
        target: Ordered sequence of data
        variant: Indication of which variant of test to be used

    Return:
        Correlation tau statistic

    Raises:
        ValueError: If ``variant`` is not from ``['a', 'b', 'c']``

    Example (single output regression):
        >>> from torchmetrics.functional.regression import kendal_rank_corrcoef
        >>> target = torch.tensor([3, -0.5, 2, 7])
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> kendal_rank_corrcoef(preds, target)

    Example (multi output regression):
        >>> from torchmetrics.functional.regression import kendal_rank_corrcoef
        >>> target = torch.tensor([[3, -0.5], [2, 7]])
        >>> preds = torch.tensor([[2.5, 0.0], [2, 8]])
        >>> kendal_rank_corrcoef(preds, target)
    """
    if variant not in ["a", "b", "c"]:
        raise ValueError(f"Argument `variant` is expected to be one of ['a', 'b', 'c'], but got {variant!r}.")
    d = preds.shape[1] if preds.ndim == 2 else 1
    _temp = torch.zeros(d, dtype=preds.dtype, device=preds.device)
    concordant_pairs, discordant_pairs = _temp.clone(), _temp.clone()
    if variant == "b":
        preds_ties, target_ties, total = _temp.clone(), _temp.clone(), _temp.clone()
    else:
        preds_ties = target_ties = total = None

    concordant_pairs, discordant_pairs, preds_ties, target_ties = _kendall_corrcoef_update(
        preds,
        target,
        concordant_pairs,
        discordant_pairs,
        preds_ties,
        target_ties,
        total,
        num_outputs=1 if preds.ndim == 1 else preds.shape[-1],
        variant=variant,
    )
    return _kendall_corrcoef_compute(concordant_pairs, discordant_pairs, preds_ties, target_ties, total, variant)
