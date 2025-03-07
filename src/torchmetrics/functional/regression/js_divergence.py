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

from typing import Union

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.regression.kl_divergence import kl_divergence
from torchmetrics.utilities.checks import _check_same_shape


def _jsd_update(p: Tensor, q: Tensor, log_prob: bool) -> tuple[Tensor, int]:
    """Update and returns jensen-shannon divergence scores for each observation and the total number of observations.

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
        mean = torch.logsumexp(torch.stack([p, q]), dim=0) - torch.log(torch.tensor(2.0))
        measures = 0.5 * kl_divergence(p, mean, log_prob=log_prob, reduction=None) + 0.5 * kl_divergence(
            q, mean, log_prob=log_prob, reduction=None
        )
    else:
        p = p / p.sum(axis=-1, keepdim=True)  # type: ignore[call-overload]
        q = q / q.sum(axis=-1, keepdim=True)  # type: ignore[call-overload]
        mean = (p + q) / 2
        measures = 0.5 * kl_divergence(p, mean, log_prob=log_prob, reduction=None) + 0.5 * kl_divergence(
            q, mean, log_prob=log_prob, reduction=None
        )
    return measures, total


def _jsd_compute(
    measures: Tensor, total: Union[int, Tensor], reduction: Literal["mean", "sum", "none", None] = "mean"
) -> Tensor:
    """Compute and reduce the Jensen-Shannon divergence based on the type of reduction."""
    if reduction == "sum":
        return measures.sum()
    if reduction == "mean":
        return measures.sum() / total
    if reduction is None or reduction == "none":
        return measures
    return measures / total


def jensen_shannon_divergence(
    p: Tensor, q: Tensor, log_prob: bool = False, reduction: Literal["mean", "sum", "none", None] = "mean"
) -> Tensor:
    r"""Compute `Jensen-Shannon divergence`_.

    .. math::
        D_{JS}(P||Q) = \frac{1}{2} D_{KL}(P||M) + \frac{1}{2} D_{KL}(Q||M)

    Where :math:`P` and :math:`Q` are probability distributions where :math:`P` usually represents a distribution
    over data and :math:`Q` is often a prior or approximation of :math:`P`. :math:`D_{KL}` is the `KL divergence`_ and
    :math:`M` is the average of the two distributions. It should be noted that the Jensen-Shannon divergence is a
    symmetrical metric i.e. :math:`D_{JS}(P||Q) = D_{JS}(Q||P)`.

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
        >>> jensen_shannon_divergence(p, q)
        tensor(0.0225)

    """
    measures, total = _jsd_update(p, q, log_prob)
    return _jsd_compute(measures, total, reduction)
