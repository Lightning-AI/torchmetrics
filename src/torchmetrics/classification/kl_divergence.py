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
from typing import Any

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.classification.kl_divergence import _kld_compute, _kld_update
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat


class KLDivergence(Metric):
    r"""Computes the `KL divergence`_:

    .. math::
        D_{KL}(P||Q) = \sum_{x\in\mathcal{X}} P(x) \log\frac{P(x)}{Q{x}}

    Where :math:`P` and :math:`Q` are probability distributions where :math:`P` usually represents a distribution
    over data and :math:`Q` is often a prior or approximation of :math:`P`. It should be noted that the KL divergence
    is a non-symetrical metric i.e. :math:`D_{KL}(P||Q) \neq D_{KL}(Q||P)`.

    Args:
        p: data distribution with shape ``[N, d]``
        q: prior or approximate distribution with shape ``[N, d]``
        log_prob: bool indicating if input is log-probabilities or probabilities. If given as probabilities,
            will normalize to make sure the distributes sum to 1.
        reduction:
            Determines how to reduce over the ``N``/batch dimension:

            - ``'mean'`` [default]: Averages score across samples
            - ``'sum'``: Sum score across samples
            - ``'none'`` or ``None``: Returns score per sample

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.


    Raises:
        TypeError:
            If ``log_prob`` is not an ``bool``.
        ValueError:
            If ``reduction`` is not one of ``'mean'``, ``'sum'``, ``'none'`` or ``None``.

    .. note::
        Half precision is only support on GPU for this metric

    Example:
        >>> import torch
        >>> from torchmetrics.functional import kl_divergence
        >>> p = torch.tensor([[0.36, 0.48, 0.16]])
        >>> q = torch.tensor([[1/3, 1/3, 1/3]])
        >>> kl_divergence(p, q)
        tensor(0.0853)

    """
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    total: Tensor

    def __init__(
        self,
        log_prob: bool = False,
        reduction: Literal["mean", "sum", "none", None] = "mean",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not isinstance(log_prob, bool):
            raise TypeError(f"Expected argument `log_prob` to be bool but got {log_prob}")
        self.log_prob = log_prob

        allowed_reduction = ["mean", "sum", "none", None]
        if reduction not in allowed_reduction:
            raise ValueError(f"Expected argument `reduction` to be one of {allowed_reduction} but got {reduction}")
        self.reduction = reduction

        if self.reduction in ["mean", "sum"]:
            self.add_state("measures", torch.tensor(0.0), dist_reduce_fx="sum")
        else:
            self.add_state("measures", [], dist_reduce_fx="cat")
        self.add_state("total", torch.tensor(0), dist_reduce_fx="sum")

    def update(self, p: Tensor, q: Tensor) -> None:  # type: ignore
        measures, total = _kld_update(p, q, self.log_prob)
        if self.reduction is None or self.reduction == "none":
            self.measures.append(measures)
        else:
            self.measures += measures.sum()
            self.total += total

    def compute(self) -> Tensor:
        measures = dim_zero_cat(self.measures) if self.reduction is None or self.reduction == "none" else self.measures
        return _kld_compute(measures, self.total, self.reduction)
