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
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.regression.utils import _check_data_shape_to_num_outputs
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.data import _bincount, _cumsum, dim_zero_cat
from torchmetrics.utilities.enums import EnumStr


class _MetricVariant(EnumStr):
    """Enumerate for metric variants."""

    A = "a"
    B = "b"
    C = "c"

    @staticmethod
    def _name() -> str:
        return "variant"


class _TestAlternative(EnumStr):
    """Enumerate for test alternative options."""

    TWO_SIDED = "two-sided"
    LESS = "less"
    GREATER = "greater"

    @staticmethod
    def _name() -> str:
        return "alternative"


def _sort_on_first_sequence(x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
    """Sort sequences in an ascent order according to the sequence ``x``."""
    # We need to clone `y` tensor not to change an object in memory
    y = torch.clone(y)
    x, y = x.T, y.T
    x, perm = x.sort()
    for i in range(x.shape[0]):
        y[i] = y[i][perm[i]]
    return x.T, y.T


def _concordant_element_sum(x: Tensor, y: Tensor, i: int) -> Tensor:
    """Count a total number of concordant pairs in a single sequence."""
    return torch.logical_and(x[i] < x[(i + 1) :], y[i] < y[(i + 1) :]).sum(0).unsqueeze(0)


def _count_concordant_pairs(preds: Tensor, target: Tensor) -> Tensor:
    """Count a total number of concordant pairs in given sequences."""
    return torch.cat([_concordant_element_sum(preds, target, i) for i in range(preds.shape[0])]).sum(0)


def _discordant_element_sum(x: Tensor, y: Tensor, i: int) -> Tensor:
    """Count a total number of discordant pairs in a single sequences."""
    return (
        torch.logical_or(
            torch.logical_and(x[i] > x[(i + 1) :], y[i] < y[(i + 1) :]),
            torch.logical_and(x[i] < x[(i + 1) :], y[i] > y[(i + 1) :]),
        )
        .sum(0)
        .unsqueeze(0)
    )


def _count_discordant_pairs(preds: Tensor, target: Tensor) -> Tensor:
    """Count a total number of discordant pairs in given sequences."""
    return torch.cat([_discordant_element_sum(preds, target, i) for i in range(preds.shape[0])]).sum(0)


def _convert_sequence_to_dense_rank(x: Tensor, sort: bool = False) -> Tensor:
    """Convert a sequence to the rank tensor."""
    # Sort if a sequence has not been sorted before
    if sort:
        x = x.sort(dim=0).values
    _ones = torch.zeros(1, x.shape[1], dtype=torch.int32, device=x.device)
    return _cumsum(torch.cat([_ones, (x[1:] != x[:-1]).int()], dim=0), dim=0)


def _get_ties(x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Get a total number of ties and staistics for p-value calculation for  a given sequence."""
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
    preds: Tensor, target: Tensor, variant: _MetricVariant
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
    preds, target = _sort_on_first_sequence(preds, target)

    concordant_pairs = _count_concordant_pairs(preds, target)
    discordant_pairs = _count_discordant_pairs(preds, target)

    n_total = torch.tensor(preds.shape[0], device=preds.device)
    preds_ties = target_ties = None
    preds_ties_p1 = preds_ties_p2 = target_ties_p1 = target_ties_p2 = None
    if variant != _MetricVariant.A:
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


def _calculate_tau(
    preds: Tensor,
    target: Tensor,
    concordant_pairs: Tensor,
    discordant_pairs: Tensor,
    con_min_dis_pairs: Tensor,
    n_total: Tensor,
    preds_ties: Optional[Tensor],
    target_ties: Optional[Tensor],
    variant: _MetricVariant,
) -> Tensor:
    """Calculate Kendall's tau from metric metadata."""
    if variant == _MetricVariant.A:
        return con_min_dis_pairs / (concordant_pairs + discordant_pairs)
    if variant == _MetricVariant.B:
        total_combinations: Tensor = n_total * (n_total - 1) // 2
        denominator = (total_combinations - preds_ties) * (total_combinations - target_ties)
        return con_min_dis_pairs / torch.sqrt(denominator)

    preds_unique = torch.tensor([len(p.unique()) for p in preds.T], dtype=preds.dtype, device=preds.device)
    target_unique = torch.tensor([len(t.unique()) for t in target.T], dtype=target.dtype, device=target.device)
    min_classes = torch.minimum(preds_unique, target_unique)
    return 2 * con_min_dis_pairs / ((min_classes - 1) / min_classes * n_total**2)


def _get_p_value_for_t_value_from_dist(t_value: Tensor) -> Tensor:
    """Obtain p-value for a given Tensor of t-values. Handle ``nan`` which cannot be passed into torch distributions.

    When t-value is ``nan``, a resulted p-value should be alson ``nan``.
    """
    device = t_value
    normal_dist = torch.distributions.normal.Normal(torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))

    is_nan = t_value.isnan()
    t_value = t_value.nan_to_num()
    p_value = normal_dist.cdf(t_value)
    return p_value.where(~is_nan, torch.tensor(float("nan"), dtype=p_value.dtype, device=p_value.device))


def _calculate_p_value(
    con_min_dis_pairs: Tensor,
    n_total: Tensor,
    preds_ties: Optional[Tensor],
    preds_ties_p1: Optional[Tensor],
    preds_ties_p2: Optional[Tensor],
    target_ties: Optional[Tensor],
    target_ties_p1: Optional[Tensor],
    target_ties_p2: Optional[Tensor],
    variant: _MetricVariant,
    alternative: Optional[_TestAlternative],
) -> Tensor:
    """Calculate p-value for Kendall's tau from metric metadata."""
    t_value_denominator_base = n_total * (n_total - 1) * (2 * n_total + 5)
    if variant == _MetricVariant.A:
        t_value = 3 * con_min_dis_pairs / torch.sqrt(t_value_denominator_base / 2)
    else:
        m = n_total * (n_total - 1)
        t_value_denominator: Tensor = (t_value_denominator_base - preds_ties_p2 - target_ties_p2) / 18
        t_value_denominator += (2 * preds_ties * target_ties) / m  # type: ignore
        t_value_denominator += preds_ties_p1 * target_ties_p1 / (9 * m * (n_total - 2))  # type: ignore
        t_value = con_min_dis_pairs / torch.sqrt(t_value_denominator)

    if alternative == _TestAlternative.TWO_SIDED:
        t_value = torch.abs(t_value)
    if alternative in [_TestAlternative.TWO_SIDED, _TestAlternative.GREATER]:
        t_value *= -1
    p_value = _get_p_value_for_t_value_from_dist(t_value)
    if alternative == _TestAlternative.TWO_SIDED:
        p_value *= 2
    return p_value


def _kendall_corrcoef_update(
    preds: Tensor,
    target: Tensor,
    concat_preds: Optional[List[Tensor]] = None,
    concat_target: Optional[List[Tensor]] = None,
    num_outputs: int = 1,
) -> Tuple[List[Tensor], List[Tensor]]:
    """Update variables required to compute Kendall rank correlation coefficient.

    Args:
        preds: Sequence of data
        target: Sequence of data
        concat_preds: List of batches of preds sequence to be concatenated
        concat_target: List of batches of target sequence to be concatenated
        num_outputs: Number of outputs in multioutput setting

    Raises:
        RuntimeError: If ``preds`` and ``target`` do not have the same shape
    """
    concat_preds = concat_preds or []
    concat_target = concat_target or []
    # Data checking
    _check_same_shape(preds, target)
    _check_data_shape_to_num_outputs(preds, target, num_outputs)

    if num_outputs == 1:
        preds = preds.unsqueeze(1)
        target = target.unsqueeze(1)

    concat_preds.append(preds)
    concat_target.append(target)

    return concat_preds, concat_target


def _kendall_corrcoef_compute(
    preds: Tensor,
    target: Tensor,
    variant: _MetricVariant,
    alternative: Optional[_TestAlternative] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Compute Kendall rank correlation coefficient, and optionally p-value of corresponding statistical test.

    Args:
        Args:
        preds: Sequence of data
        target: Sequence of data
        variant: Indication of which variant of Kendall's tau to be used
        alternative: Alternative hypothesis for for t-test. Possible values:
            - 'two-sided': the rank correlation is nonzero
            - 'less': the rank correlation is negative (less than zero)
            - 'greater':  the rank correlation is positive (greater than zero)
    """
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

    tau = _calculate_tau(
        preds, target, concordant_pairs, discordant_pairs, con_min_dis_pairs, n_total, preds_ties, target_ties, variant
    )
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
    r"""Compute `Kendall Rank Correlation Coefficient`_.

    .. math::
        tau_a = \frac{C - D}{C + D}

    where :math:`C` represents concordant pairs, :math:`D` stands for discordant pairs.

    .. math::
        tau_b = \frac{C - D}{\sqrt{(C + D + T_{preds}) * (C + D + T_{target})}}

    where :math:`C` represents concordant pairs, :math:`D` stands for discordant pairs and :math:`T` represents
    a total number of ties.

    .. math::
        tau_c = 2 * \frac{C - D}{n^2 * \frac{m - 1}{m}}

    where :math:`C` represents concordant pairs, :math:`D` stands for discordant pairs, :math:`n` is a total number
    of observations and :math:`m` is a ``min`` of unique values in ``preds`` and ``target`` sequence.

    Definitions according to Definition according to `The Treatment of Ties in Ranking Problems`_.

    Args:
        preds: Sequence of data of either shape ``(N,)`` or ``(N,d)``
        target: Sequence of data of either shape ``(N,)`` or ``(N,d)``
        variant: Indication of which variant of Kendall's tau to be used
        t_test: Indication whether to run t-test
        alternative: Alternative hypothesis for t-test. Possible values:
            - 'two-sided': the rank correlation is nonzero
            - 'less': the rank correlation is negative (less than zero)
            - 'greater':  the rank correlation is positive (greater than zero)

    Return:
        Correlation tau statistic
        (Optional) p-value of corresponding statistical test (asymptotic)

    Raises:
        ValueError: If ``t_test`` is not of a type bool
        ValueError: If ``t_test=True`` and ``alternative=None``

    Example (single output regression):
        >>> from torchmetrics.functional.regression import kendall_rank_corrcoef
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> target = torch.tensor([3, -0.5, 2, 1])
        >>> kendall_rank_corrcoef(preds, target)
        tensor(0.3333)

    Example (multi output regression):
        >>> from torchmetrics.functional.regression import kendall_rank_corrcoef
        >>> preds = torch.tensor([[2.5, 0.0], [2, 8]])
        >>> target = torch.tensor([[3, -0.5], [2, 1]])
        >>> kendall_rank_corrcoef(preds, target)
        tensor([1., 1.])

    Example (single output regression with t-test)
        >>> from torchmetrics.functional.regression import kendall_rank_corrcoef
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> target = torch.tensor([3, -0.5, 2, 1])
        >>> kendall_rank_corrcoef(preds, target, t_test=True, alternative='two-sided')
        (tensor(0.3333), tensor(0.4969))

    Example (multi output regression with t-test):
        >>> from torchmetrics.functional.regression import kendall_rank_corrcoef
        >>> preds = torch.tensor([[2.5, 0.0], [2, 8]])
        >>> target = torch.tensor([[3, -0.5], [2, 1]])
        >>> kendall_rank_corrcoef(preds, target, t_test=True, alternative='two-sided')
            (tensor([1., 1.]), tensor([nan, nan]))
    """
    if not isinstance(t_test, bool):
        raise ValueError(f"Argument `t_test` is expected to be of a type `bool`, but got {type(t_test)}.")
    if t_test and alternative is None:
        raise ValueError("Argument `alternative` is required if `t_test=True` but got `None`.")

    _variant = _MetricVariant.from_str(str(variant))
    _alternative = _TestAlternative.from_str(str(alternative)) if t_test else None

    _preds, _target = _kendall_corrcoef_update(
        preds, target, [], [], num_outputs=1 if preds.ndim == 1 else preds.shape[-1]
    )
    tau, p_value = _kendall_corrcoef_compute(
        dim_zero_cat(_preds), dim_zero_cat(_target), _variant, _alternative  # type: ignore[arg-type]  # todo
    )

    if p_value is not None:
        return tau, p_value
    return tau
