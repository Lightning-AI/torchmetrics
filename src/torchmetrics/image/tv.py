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
from typing import Any, List, Optional, Sequence, Union

import torch
from torch import Tensor, tensor
from typing_extensions import Literal

from torchmetrics.functional.image.tv import _total_variation_compute, _total_variation_update
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["TotalVariation.plot"]


class TotalVariation(Metric):
    """Compute Total Variation loss (`TV`_).

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``img`` (:class:`~torch.Tensor`): A tensor of shape ``(N, C, H, W)`` consisting of images

    As output of `forward` and `compute` the metric returns the following output

    - ``sdi`` (:class:`~torch.Tensor`): if ``reduction!='none'`` returns float scalar tensor with average TV value
      over sample else returns tensor of shape ``(N,)`` with TV values per sample

    Args:
        reduction: a method to reduce metric score over samples

            - ``'mean'``: takes the mean over samples
            - ``'sum'``: takes the sum over samples
            - ``None`` or ``'none'``: return the score per sample

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``reduction`` is not one of ``'sum'``, ``'mean'``, ``'none'`` or ``None``

    Example:
        >>> import torch
        >>> from torchmetrics.image import TotalVariation
        >>> _ = torch.manual_seed(42)
        >>> tv = TotalVariation()
        >>> img = torch.rand(5, 3, 28, 28)
        >>> tv(img)
        tensor(7546.8018)
    """

    full_state_update: bool = False
    is_differentiable: bool = True
    higher_is_better: bool = False
    plot_lower_bound: float = 0.0

    num_elements: Tensor
    score_list: List[Tensor]
    score: Tensor

    def __init__(self, reduction: Optional[Literal["mean", "sum", "none"]] = "sum", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if reduction is not None and reduction not in ("sum", "mean", "none"):
            raise ValueError("Expected argument `reduction` to either be 'sum', 'mean', 'none' or None")
        self.reduction = reduction

        self.add_state("score_list", default=[], dist_reduce_fx="cat")
        self.add_state("score", default=tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("num_elements", default=tensor(0, dtype=torch.int), dist_reduce_fx="sum")

    def update(self, img: Tensor) -> None:
        """Update current score with batch of input images."""
        score, num_elements = _total_variation_update(img)
        if self.reduction is None or self.reduction == "none":
            self.score_list.append(score)
        else:
            self.score += score.sum()
        self.num_elements += num_elements

    def compute(self) -> Tensor:
        """Compute final total variation."""
        score = dim_zero_cat(self.score_list) if self.reduction is None or self.reduction == "none" else self.score
        return _total_variation_compute(score, self.num_elements, self.reduction)

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> import torch
            >>> from torchmetrics.image import TotalVariation
            >>> metric = TotalVariation()
            >>> metric.update(torch.rand(5, 3, 28, 28))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.image import TotalVariation
            >>> metric = TotalVariation()
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(torch.rand(5, 3, 28, 28)))
            >>> fig_, ax_ = metric.plot(values)
        """
        return self._plot(val, ax)
