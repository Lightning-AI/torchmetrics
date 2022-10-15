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

from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.data import _bincount
from torchmetrics.utilities.enums import EnumStr


def _dim_one_cat(x: Union[Tensor, List[Tensor]]) -> Tensor:
    """Concatenation along the one dimension."""
    x = x if isinstance(x, (list, tuple)) else [x]
    x = [y.unsqueeze(0) if y.numel() == 1 and y.ndim == 0 else y for y in x]
    if not x:  # empty list
        raise ValueError("No samples to concatenate")
    return torch.cat(x, dim=1)


class _TestAlternative(EnumStr):
    TWO_SIDED = "two-sided"
    LESS = "less"
    GREATER = "greater"

    @classmethod
    def from_str(cls, value: str) -> Optional["EnumStr"]:
        """
        Raises:
            ValueError:
                If required test alternativeis not among the supported options.
        """
        _allowed_alternative = [im.lower().replace("_", "-") for im in _TestAlternative._member_names_]

        enum_key = super().from_str(value.replace("-", "_"))
        if enum_key is not None and enum_key in _allowed_alternative:
            return enum_key
        raise ValueError(f"Invalid test alternative. Expected one of {_allowed_alternative}, but got {enum_key}.")


def _sort_on_first_sequence(x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
    """Sort sequences in an ascent order according to the sequence ``x``."""
    # We need to clone `y` tensor not to change it in memory
    y = torch.clone(y)
    x, perm = x.sort()
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


def _convert_sequence_to_dense_rank(x: Tensor, sort: bool = False) -> Tensor:
    """Convert a sequence to the rank tensor."""
    # Sort if a sequence has not been sorted before
    if sort:
        x = x.sort(dim=0).values
    _ones = torch.zeros(1, x.shape[1], dtype=torch.int32, device=x.device)
    return torch.cat([_ones, (x[1:] != x[:-1]).int()], dim=0).cumsum(0)


def _get_ties(x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Get number of ties and staistics for p-value calculation for  a given sequence."""
    ties = torch.zeros(x.shape[1], dtype=x.dtype, device=x.device)
    ties_p1 = torch.zeros(x.shape[1], dtype=x.dtype, device=x.device)
    ties_p2 = torch.zeros(x.shape[1], dtype=x.dtype, device=x.device)
    for dim in range(x.shape[1]):
        n_ties = _bincount(x[:, dim])
        n_ties = n_ties[n_ties > 1]
        ties[dim] = (n_ties * (n_ties - 1) // 2).sum()
        ties_p1[dim] = (n_ties * (n_ties - 1.0) * (n_ties - 2)).sum()
        ties_p2[dim] = (n_ties * (n_ties - 1.0) * (2 * n_ties + 5)).sum()

    return ties, ties_p1, ties_p2


def _get_metric_metadata(
    preds: Tensor, target: Tensor, variant: Literal["a", "b", "c"]
) -> Tuple[
    Tensor,
    Tensor,
    Optional[Tensor],
    Optional[Tensor],
    Optional[Tensor],
    Optional[Tensor],
    Optional[Tensor],
    Optional[Tensor],
    Tensor,
]:
    """Obtain statistics to calculate metric value."""
    # Sort on target and convert it to dense rank
    preds, target = _sort_on_first_sequence(preds, target)
    preds, target = preds.T, target.T

    concordant_pairs = _count_concordant_pairs(preds, target)
    discordant_pairs = _count_discordant_pairs(preds, target)

    n_total = torch.tensor(preds.shape[0], device=preds.device)
    preds_ties = target_ties = None
    preds_ties_p1 = preds_ties_p2 = target_ties_p1 = target_ties_p2 = None
    if variant == "b":
        preds = _convert_sequence_to_dense_rank(preds)
        target = _convert_sequence_to_dense_rank(target, sort=True)
        preds_ties, preds_ties_p1, preds_ties_p2 = _get_ties(preds)
        target_ties, target_ties_p1, target_ties_p2 = _get_ties(target)

    return (
        concordant_pairs,
        discordant_pairs,
        preds_ties,
        preds_ties_p1,
        preds_ties_p2,
        target_ties,
        target_ties_p1,
        target_ties_p2,
        n_total,
    )


def _calculate_p_value(
    con_min_dis_pairs: Tensor,
    n_total: Tensor,
    preds_ties: Optional[Tensor],
    preds_ties_p1: Optional[Tensor],
    preds_ties_p2: Optional[Tensor],
    target_ties: Optional[Tensor],
    target_ties_p1: Optional[Tensor],
    target_ties_p2: Optional[Tensor],
    variant: Literal["a", "b", "c"],
    alternative: _TestAlternative,
) -> Tensor:
    normal_dist = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))

    t_value_denominator_base = n_total * (n_total - 1) * (2 * n_total + 5)
    if variant == "a":
        t_value = 3 * con_min_dis_pairs / torch.sqrt(t_value_denominator_base / 2)
    else:
        m = n_total * (n_total - 1)
        t_value_denominator: Tensor = (t_value_denominator_base - preds_ties_p2 - target_ties_p2) / 18
        t_value_denominator += (2 * preds_ties * target_ties) / m
        t_value_denominator += preds_ties_p1 * target_ties_p1 / (9 * m * (n_total - 2))
        t_value = con_min_dis_pairs / torch.sqrt(t_value_denominator)

    if alternative in [_TestAlternative.TWO_SIDED, _TestAlternative.GREATER]:
        t_value *= -1
    p_value = normal_dist.cdf(t_value)
    if alternative == _TestAlternative.TWO_SIDED:
        p_value *= 2
    return p_value


def _kendall_corrcoef_update(
    preds: Tensor,
    target: Tensor,
    concat_preds: List[Tensor] = [],
    concat_target: List[Tensor] = [],
    num_outputs: int = 1,
) -> Tuple[List[Tensor], List[Tensor]]:
    """Update variables required to compute Kendall rank correlation coefficient.

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
    if (num_outputs == 1 and preds.ndim != 1) or (num_outputs > 1 and num_outputs != preds.shape[1]):
        raise ValueError(
            f"Expected argument `num_outputs` to match the second dimension of input, but got {num_outputs}"
            f" and {preds.shape[1]}."
        )
    if num_outputs == 1:
        preds = preds.unsqueeze(0)
        target = target.unsqueeze(0)

    concat_preds.append(preds)
    concat_target.append(target)

    return concat_preds, concat_target


def _kendall_corrcoef_compute(
    preds: Tensor,
    target: Tensor,
    variant: Literal["a", "b", "c"],
    alternative: Optional[_TestAlternative] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Compute Kendall rank correlation coefficient, and optionally p-value of corresponding statistical test."""
    (
        concordant_pairs,
        discordant_pairs,
        preds_ties,
        preds_ties_p1,
        preds_ties_p2,
        target_ties,
        target_ties_p1,
        target_ties_p2,
        n_total,
    ) = _get_metric_metadata(preds, target, variant)
    con_min_dis_pairs = concordant_pairs - discordant_pairs

    if variant == "a":
        tau = con_min_dis_pairs / (concordant_pairs + discordant_pairs)
    elif variant == "b":
        total_combinations: Tensor = n_total * (n_total - 1) // 2
        denominator = (total_combinations - preds_ties) * (total_combinations - target_ties)
        tau = con_min_dis_pairs / torch.sqrt(denominator)
    else:
        tau = 2 * con_min_dis_pairs / (n_total**2)

    p_value = (
        _calculate_p_value(
            con_min_dis_pairs,
            n_total,
            preds_ties,
            preds_ties_p1,
            preds_ties_p2,
            target_ties,
            target_ties_p1,
            target_ties_p2,
            variant,
            alternative,
        )
        if alternative
        else None
    )

    # Squeeze tensor if num_outputs=1
    if tau.shape[0] == 1:
        tau = tau.squeeze()
        p_value = p_value.squeeze() if p_value is not None else None

    return tau.clamp(-1, 1), p_value


def kendall_rank_corrcoef(
    preds: Tensor,
    target: Tensor,
    variant: Literal["a", "b", "c"] = "b",
    t_test: bool = False,
    alternative: Optional[Literal["two-sided", "less", "greater"]] = "two-sided",
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Computes `Kendall Rank Correlation Coefficient`_.

    Args:
        preds: Ordered sequence of data
        target: Ordered sequence of data
        variant: Indication of which variant of Kendall's tau to be used
        t_test: Indication whether to run t-test
        alternative: Alternative hypothesis for for t-test. Possible values:
            - 'two-sided': the rank correlation is nonzero
            - 'less': the rank correlation is negative (less than zero)
            - 'greater':  the rank correlation is positive (greater than zero)

    Return:
        Correlation tau statistic
        (Optional) p-value of corresponding statistical test (asymptotic)

    Raises:
        ValueError: If ``variant`` is not from ``['a', 'b', 'c']``

    Example (single output regression):
        >>> from torchmetrics.functional.regression import kendall_rank_corrcoef
        >>> target = torch.tensor([3, -0.5, 2, 1])
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> kendall_rank_corrcoef(preds, target)
        tensor(0.3333)

    Example (multi output regression):
        >>> from torchmetrics.functional.regression import kendall_rank_corrcoef
        >>> target = torch.tensor([[3, -0.5], [2, 1]])
        >>> preds = torch.tensor([[2.5, 0.0], [2, 8]])
        >>> kendall_rank_corrcoef(preds, target)
        tensor([ 1., -1.])
    """
    if variant not in ["a", "b", "c"]:
        raise ValueError(f"Argument `variant` is expected to be one of `['a', 'b', 'c']`, but got {variant!r}.")
    if not isinstance(t_test, bool):
        raise ValueError(f"Argument `t_test` is expected to be of a type `bool`, but got {type(t_test)}.")
    _alternative = _TestAlternative.from_str(alternative) if t_test else None

    _preds, _target = _kendall_corrcoef_update(
        preds, target, [], [], num_outputs=1 if preds.ndim == 1 else preds.shape[-1]
    )
    print(_preds[0].shape)
    tau, p_value = _kendall_corrcoef_compute(_dim_one_cat(_preds), _dim_one_cat(_target), variant, _alternative)

    if p_value is not None:
        return tau, p_value
    return tau
