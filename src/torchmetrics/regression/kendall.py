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

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.regression.kendall import (
    _dim_one_cat,
    _kendall_corrcoef_compute,
    _kendall_corrcoef_update,
    _TestAlternative,
)
from torchmetrics.metric import Metric


class KendallRankCorrCoef(Metric):
    r"""Computes `Kendall Rank Correlation Coefficient`_:

    Where :math:`y` is a tensor of target values, and :math:`x` is a tensor of predictions.

    Forward accepts

    - ``preds``: Ordered sequence of data
    - ``target``: Ordered sequence of data

    Args:
        variant: Indication of which variant of Kendall's tau to be used
        t_test: Indication whether to run t-test
        alternative: Alternative hypothesis for for t-test. Possible values:
            - 'two-sided': the rank correlation is nonzero
            - 'less': the rank correlation is negative (less than zero)
            - 'greater':  the rank correlation is positive (greater than zero)
        num_outputs: Number of outputs in multioutput setting
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (single output regression):
        >>> from torchmetrics.regression import KendallRankCorrCoef
        >>> target = torch.tensor([3, -0.5, 2, 1])
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> kendall = KendallRankCorrCoef()
        >>> kendall(preds, target)
        tensor(0.3333)

    Example (multi output regression):
        kendall
        >>> target = torch.tensor([[3, -0.5], [2, 1]])
        >>> preds = torch.tensor([[2.5, 0.0], [2, 8]])
        >>> kendall = KendallRankCorrCoef()
        >>> kendall(preds, target)
        tensor([ 1., -1.])
    """

    is_differentiable = True
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
        if variant not in ["a", "b", "c"]:
            raise ValueError(f"Argument `variant` is expected to be one of ['a', 'b', 'c'], but got {variant!r}.")
        self.variant = variant
        if not isinstance(t_test, bool):
            raise ValueError(f"Argument `t_test` is expected to be of a type `bool`, but got {type(t_test)}.")
        self.alternative = _TestAlternative.from_str(alternative) if t_test else None
        self.num_outputs = num_outputs

        self.add_state("preds", [], dist_reduce_fx="cat")
        self.add_state("target", [], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update variables required to compute Kendall rank correlation coefficient.

        Args:
            preds: Ordered sequence of data
            target: Ordered sequence of data
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
        preds = _dim_one_cat(self.preds)
        target = _dim_one_cat(self.target)
        tau, p_value = _kendall_corrcoef_compute(preds, target, self.variant, self.alternative)

        if p_value is not None:
            return tau, p_value
        return tau
