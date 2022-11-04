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

from typing import Any, List, Optional, Tuple, Union

from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.regression.kendall import (
    _kendall_corrcoef_compute,
    _kendall_corrcoef_update,
    _MetricVariant,
    _TestAlternative,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat


class KendallRankCorrCoef(Metric):
    r"""Computes `Kendall Rank Correlation Coefficient`_:

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

    Forward accepts

    - ``preds`` (float tensor): Sequence of data of either shape ``(N,)`` or ``(N,d)``
    - ``target`` (float tensor): Sequence of data of either shape ``(N,)`` or ``(N,d)``

    Args:
        variant: Indication of which variant of Kendall's tau to be used
        t_test: Indication whether to run t-test
        alternative: Alternative hypothesis for t-test. Possible values:
            - 'two-sided': the rank correlation is nonzero
            - 'less': the rank correlation is negative (less than zero)
            - 'greater':  the rank correlation is positive (greater than zero)
        num_outputs: Number of outputs in multioutput setting
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError: If ``t_test`` is not of a type bool
        ValueError: If ``t_test=True`` and ``alternative=None``

    Example (single output regression):
        >>> import torch
        >>> from torchmetrics.regression import KendallRankCorrCoef
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> target = torch.tensor([3, -0.5, 2, 1])
        >>> kendall = KendallRankCorrCoef()
        >>> kendall(preds, target)
        tensor(0.3333)

    Example (multi output regression):
        >>> import torch
        >>> from torchmetrics.regression import KendallRankCorrCoef
        >>> preds = torch.tensor([[2.5, 0.0], [2, 8]])
        >>> target = torch.tensor([[3, -0.5], [2, 1]])
        >>> kendall = KendallRankCorrCoef(num_outputs=2)
        >>> kendall(preds, target)
        tensor([1., 1.])

    Example (single output regression with t-test):
        >>> import torch
        >>> from torchmetrics.regression import KendallRankCorrCoef
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> target = torch.tensor([3, -0.5, 2, 1])
        >>> kendall = KendallRankCorrCoef(t_test=True, alternative='two-sided')
        >>> kendall(preds, target)
        (tensor(0.3333), tensor(0.4969))

    Example (multi output regression with t-test):
        >>> import torch
        >>> from torchmetrics.regression import KendallRankCorrCoef
        >>> preds = torch.tensor([[2.5, 0.0], [2, 8]])
        >>> target = torch.tensor([[3, -0.5], [2, 1]])
        >>> kendall = KendallRankCorrCoef(t_test=True, alternative='two-sided', num_outputs=2)
        >>> kendall(preds, target)
        (tensor([1., 1.]), tensor([nan, nan]))
    """

    is_differentiable = False
    higher_is_better = None
    full_state_update = True
    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
        self,
        variant: Literal["a", "b", "c"] = "b",
        t_test: bool = False,
        alternative: Optional[Literal["two-sided", "less", "greater"]] = "two-sided",
        num_outputs: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if not isinstance(t_test, bool):
            raise ValueError(f"Argument `t_test` is expected to be of a type `bool`, but got {type(t_test)}.")
        if t_test and alternative is None:
            raise ValueError("Argument `alternative` is required if `t_test=True` but got `None`.")

        self.variant = _MetricVariant.from_str(variant)
        self.alternative = _TestAlternative.from_str(alternative) if t_test and alternative else None
        self.num_outputs = num_outputs

        self.add_state("preds", [], dist_reduce_fx="cat")
        self.add_state("target", [], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update variables required to compute Kendall rank correlation coefficient.

        Args:
            preds: Sequence of data of either shape ``(N,)`` or ``(N,d)``
            target: Sequence of data of either shape ``(N,)`` or ``(N,d)``
        """
        self.preds, self.target = _kendall_corrcoef_update(
            preds,
            target,
            self.preds,
            self.target,
            num_outputs=self.num_outputs,
        )

    def compute(self) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Compute Kendall rank correlation coefficient, and optionally p-value of corresponding statistical test.

        Return:
            Correlation tau statistic
            (Optional) p-value of corresponding statistical test (asymptotic)
        """
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        tau, p_value = _kendall_corrcoef_compute(preds, target, self.variant, self.alternative)

        if p_value is not None:
            return tau, p_value
        return tau
