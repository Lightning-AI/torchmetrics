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

from typing import Tuple, Union

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.compute import _safe_xlogy


def _kld_update(p: Tensor, q: Tensor, log_prob: bool) -> Tuple[Tensor, int]:
    """Update and returns KL divergence scores for each observation and the total number of observations.

    Args:
        p: data distribution with shape ``[N, d]``
        q: prior or approximate distribution with shape ``[N, d]``
        log_prob: bool indicating if input is log-probabilities or probabilities. If given as probabilities,
            will normalize to make sure the distributes sum to 1
    """
    _check_same_shape(p, q)
    if p.ndim != 2 or q.ndim != 2:
        raise ValueError(f"Expected both p and q distribution to be 2D but got {p.ndim} and {q.ndim} respectively")

    total = p.shape[0]
    if log_prob:
        measures = torch.sum(p.exp() * (p - q), axis=-1)  # type: ignore[call-overload]
    else:
        p = p / p.sum(axis=-1, keepdim=True)  # type: ignore[call-overload]
        q = q / q.sum(axis=-1, keepdim=True)  # type: ignore[call-overload]
        measures = _safe_xlogy(p, p / q).sum(axis=-1)  # type: ignore[call-overload]

    return measures, total


def _kld_compute(
    measures: Tensor, total: Union[int, Tensor], reduction: Literal["mean", "sum", "none", None] = "mean"
) -> Tensor:
    """Compute the KL divergenece based on the type of reduction.

    Args:
        measures: Tensor of KL divergence scores for each observation
        total: Number of observations
        reduction:
            Determines how to reduce over the ``N``/batch dimension:

            - ``'mean'`` [default]: Averages score across samples
            - ``'sum'``: Sum score across samples
            - ``'none'`` or ``None``: Returns score per sample

    Example:
        >>> p = torch.tensor([[0.36, 0.48, 0.16]])
        >>> q = torch.tensor([[1/3, 1/3, 1/3]])
        >>> measures, total = _kld_update(p, q, log_prob=False)
        >>> _kld_compute(measures, total)
        tensor(0.0853)
    """
    if reduction == "sum":
        return measures.sum()
    if reduction == "mean":
        return measures.sum() / total
    if reduction is None or reduction == "none":
        return measures
    return measures / total


def kl_divergence(
    p: Tensor, q: Tensor, log_prob: bool = False, reduction: Literal["mean", "sum", "none", None] = "mean"
) -> Tensor:
    r"""Compute `KL divergence`_.

    .. math::
        D_{KL}(P||Q) = \sum_{x\in\mathcal{X}} P(x) \log\frac{P(x)}{Q{x}}

    Where :math:`P` and :math:`Q` are probability distributions where :math:`P` usually represents a distribution
    over data and :math:`Q` is often a prior or approximation of :math:`P`. It should be noted that the KL divergence
    is a non-symetrical metric i.e. :math:`D_{KL}(P||Q) \neq D_{KL}(Q||P)`.

    Args:
        p: data distribution with shape ``[N, d]``
        q: prior or approximate distribution with shape ``[N, d]``
        log_prob: bool indicating if input is log-probabilities or probabilities. If given as probabilities,
            will normalize to make sure the distributes sum to 1
        reduction:
            Determines how to reduce over the ``N``/batch dimension:

            - ``'mean'`` [default]: Averages score across samples
            - ``'sum'``: Sum score across samples
            - ``'none'`` or ``None``: Returns score per sample

    Example:
        >>> from torch import tensor
        >>> p = tensor([[0.36, 0.48, 0.16]])
        >>> q = tensor([[1/3, 1/3, 1/3]])
        >>> kl_divergence(p, q)
        tensor(0.0853)
    """
    measures, total = _kld_update(p, q, log_prob)
    return _kld_compute(measures, total, reduction)
