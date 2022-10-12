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
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.regression.kl_divergence import kl_divergence as _kl_divergence
from torchmetrics.utilities.prints import rank_zero_warn


def kl_divergence(
    p: Tensor, q: Tensor, log_prob: bool = False, reduction: Literal["mean", "sum", "none", None] = "mean"
) -> Tensor:
    r"""Computes `KL divergence`_

    .. math::
        D_{KL}(P||Q) = \sum_{x\in\mathcal{X}} P(x) \log\frac{P(x)}{Q{x}}

    Where :math:`P` and :math:`Q` are probability distributions where :math:`P` usually represents a distribution
    over data and :math:`Q` is often a prior or approximation of :math:`P`. It should be noted that the KL divergence
    is a non-symetrical metric i.e. :math:`D_{KL}(P||Q) \neq D_{KL}(Q||P)`.

    .. note::
        This metric have been moved to the regression package in v0.10 and this version will be removed in v0.11.

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
        >>> import torch
        >>> p = torch.tensor([[0.36, 0.48, 0.16]])
        >>> q = torch.tensor([[1/3, 1/3, 1/3]])
        >>> kl_divergence(p, q)
        tensor(0.0853)
    """
    rank_zero_warn(
        "`torchmetrics.functional.classification.kl_divergence` have been moved to"
        "`torchmetrics.functional.regression.kl_divergence` from v0.10 and this version will be removed in v0.11."
        "Please update import paths.",
        DeprecationWarning,
    )
    return _kl_divergence(p, q, log_prob, reduction)
