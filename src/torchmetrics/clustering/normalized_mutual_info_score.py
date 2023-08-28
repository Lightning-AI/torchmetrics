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
from typing import Any, List, Literal

from torch import Tensor
from torchmetrics.clustering.mutual_info_score import MutualInfoScore

from torchmetrics.functional.clustering.normalized_mutual_info_score import normalized_mutual_info_score
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["NormalizedMutualInfoScore.plot"]


class NormalizedMutualInfoScore(MutualInfoScore):
    r"""Compute `Normalized Mutual Information Score`_.

    .. math::
        MI(U,V) = \sum_{i=1}^{\abs{U}} \sum_{j=1}^{\abs{V}} \frac{\abs{U_i\cap V_j}}{N}
        \log\frac{N\abs{U_i\cap V_j}}{\abs{U_i}\abs{V_j}}

    Where :math:`U` is a tensor of target values, :math:`V` is a tensor of predictions,
    :math:`\abs{U_i}` is the number of samples in cluster :math:`U_i`, and
    :math:`\abs{V_i}` is the number of samples in cluster :math:`V_i`.

    The metric is symmetric, therefore swapping :math:`U` and :math:`V` yields
    the same mutual information score.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): either single output float tensor with shape ``(N,)``
    - ``target`` (:class:`~torch.Tensor`): either single output tensor with shape ``(N,)``

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``mi_score`` (:class:`~torch.Tensor`): A tensor with the Mutual Information Score

    Args:
        normalization_method: generalized mean method used to calculate normalization
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> import torch
        >>> from torchmetrics.clustering import NormalizedMutualInfoScore
        >>> preds = torch.tensor([2, 1, 0, 1, 0])
        >>> target = torch.tensor([0, 2, 1, 1, 0])
        >>> nmi_score = NormalizedMutualInfoScore()
        >>> nmi_score(preds, target)
        tensor(0.5004)

    """

    is_differentiable = True
    higher_is_better = None
    full_state_update: bool = True
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 0.0
    preds: List[Tensor]
    target: List[Tensor]
    contingency: Tensor

    def __init__(
        self,
        normalization_method: Literal["min", "geometric", "arithmetic", "max"],
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.normalization_method = normalization_method

    def compute(self) -> Tensor:
        """Compute normalized mutual information over state."""
        return normalized_mutual_info_score(
            dim_zero_cat(self.preds),
            dim_zero_cat(self.target),
            self.normalization_method
        )
