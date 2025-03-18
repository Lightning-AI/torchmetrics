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
from collections.abc import Sequence
from typing import Any, Optional, Union

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.segmentation.generalized_dice import (
    _generalized_dice_compute,
    _generalized_dice_update,
    _generalized_dice_validate_args,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["GeneralizedDiceScore.plot"]


class GeneralizedDiceScore(Metric):
    r"""Compute `Generalized Dice Score`_.

    The metric can be used to evaluate the performance of image segmentation models. The Generalized Dice Score is
    defined as:

    .. math::
        GDS = \frac{2 \\sum_{i=1}^{N} w_i \\sum_{j} t_{ij} p_{ij}}{
            \\sum_{i=1}^{N} w_i \\sum_{j} t_{ij} + \\sum_{i=1}^{N} w_i \\sum_{j} p_{ij}}

    where :math:`N` is the number of classes, :math:`t_{ij}` is the target tensor, :math:`p_{ij}` is the prediction
    tensor, and :math:`w_i` is the weight for class :math:`i`. The weight can be computed in three different ways:

    - `square`: :math:`w_i = 1 / (\\sum_{j} t_{ij})^2`
    - `simple`: :math:`w_i = 1 / \\sum_{j} t_{ij}`
    - `linear`: :math:`w_i = 1`

    Note that the generalized dice loss can be computed as one minus the generalized dice score.

    As input to ``forward`` and ``update`` the metric accepts the following input:

        - ``preds`` (:class:`~torch.Tensor`): An one-hot boolean tensor of shape ``(N, C, ...)`` with ``N`` being
          the number of samples and ``C`` the number of classes. Alternatively, an integer tensor of shape ``(N, ...)``
          can be provided, where the integer values correspond to the class index. The input type can be controlled
          with the ``input_format`` argument.
        - ``target`` (:class:`~torch.Tensor`): An one-hot boolean tensor of shape ``(N, C, ...)`` with ``N`` being
          the number of samples and ``C`` the number of classes. Alternatively, an integer tensor of shape ``(N, ...)``
          can be provided, where the integer values correspond to the class index. The input type can be controlled
          with the ``input_format`` argument.

    As output to ``forward`` and ``compute`` the metric returns the following output:

        - ``gds`` (:class:`~torch.Tensor`): The generalized dice score. If ``per_class`` is set to ``True``, the output
          will be a tensor of shape ``(C,)`` with the generalized dice score for each class. If ``per_class`` is
          set to ``False``, the output will be a scalar tensor.

    Args:
        num_classes: The number of classes in the segmentation problem.
        include_background: Whether to include the background class in the computation
        per_class: Whether to compute the metric for each class separately.
        weight_type: The type of weight to apply to each class. Can be one of ``"square"``, ``"simple"``, or
            ``"linear"``.
        input_format: What kind of input the function receives. Choose between ``"one-hot"`` for one-hot encoded tensors
            or ``"index"`` for index tensors
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``num_classes`` is not a positive integer
        ValueError:
            If ``include_background`` is not a boolean
        ValueError:
            If ``per_class`` is not a boolean
        ValueError:
            If ``weight_type`` is not one of ``"square"``, ``"simple"``, or ``"linear"``
        ValueError:
            If ``input_format`` is not one of ``"one-hot"`` or ``"index"``

    Example:
        >>> from torch import randint
        >>> from torchmetrics.segmentation import GeneralizedDiceScore
        >>> gds = GeneralizedDiceScore(num_classes=3)
        >>> preds = randint(0, 2, (10, 3, 128, 128))
        >>> target = randint(0, 2, (10, 3, 128, 128))
        >>> gds(preds, target)
        tensor(0.4992)
        >>> gds = GeneralizedDiceScore(num_classes=3, per_class=True)
        >>> gds(preds, target)
        tensor([0.5001, 0.4993, 0.4982])
        >>> gds = GeneralizedDiceScore(num_classes=3, per_class=True, include_background=False)
        >>> gds(preds, target)
        tensor([0.4993, 0.4982])

    """

    score: Tensor
    samples: Tensor
    full_state_update: bool = False
    is_differentiable: bool = False
    higher_is_better: bool = True
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def __init__(
        self,
        num_classes: int,
        include_background: bool = True,
        per_class: bool = False,
        weight_type: Literal["square", "simple", "linear"] = "square",
        input_format: Literal["one-hot", "index"] = "one-hot",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        _generalized_dice_validate_args(num_classes, include_background, per_class, weight_type, input_format)
        self.num_classes = num_classes
        self.include_background = include_background
        self.per_class = per_class
        self.weight_type = weight_type
        self.input_format = input_format

        num_classes = num_classes - 1 if not include_background else num_classes
        self.add_state("score", default=torch.zeros(num_classes if per_class else 1), dist_reduce_fx="sum")
        self.add_state("samples", default=torch.zeros(1), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update the state with new data."""
        numerator, denominator = _generalized_dice_update(
            preds, target, self.num_classes, self.include_background, self.weight_type, self.input_format
        )
        self.score += _generalized_dice_compute(numerator, denominator, self.per_class).sum(dim=0)
        self.samples += preds.shape[0]

    def compute(self) -> Tensor:
        """Compute the final generalized dice score."""
        return self.score / self.samples

    def plot(self, val: Union[Tensor, Sequence[Tensor], None] = None, ax: Optional[_AX_TYPE] = None) -> _PLOT_OUT_TYPE:
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
            >>> from torchmetrics.segmentation import GeneralizedDiceScore
            >>> metric = GeneralizedDiceScore(num_classes=3)
            >>> metric.update(torch.randint(0, 2, (10, 3, 128, 128)), torch.randint(0, 2, (10, 3, 128, 128)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.segmentation import GeneralizedDiceScore
            >>> metric = GeneralizedDiceScore(num_classes=3)
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(
            ...        metric(torch.randint(0, 2, (10, 3, 128, 128)), torch.randint(0, 2, (10, 3, 128, 128)))
            ...     )
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)
