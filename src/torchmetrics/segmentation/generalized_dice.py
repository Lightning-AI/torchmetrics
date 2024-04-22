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
from typing import Any, Optional, Sequence, Union
from typing_extensions import Literal
import torch
from torch import Tensor

from torchmetrics.functional.segmentation.generalized_dice import (
    _generalized_dice_validate_args, _generalized_dice_compute, _generalized_dice_update
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["GeneralizedDiceScore.plot"]


class GeneralizedDice(Metric):
    """
    """

    score: Tensor
    num_batches: Tensor
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
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        _generalized_dice_validate_args(num_classes, include_background, per_class, weight_type)
        self.num_classes = num_classes
        self.include_background = include_background
        self.per_class = per_class
        self.weight_type = weight_type

        self.add_state("score", default=torch.zeros(num_classes if per_class else 1), dist_reduce_fx="mean")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """ Update the state with new data."""
        numerator, denominator = _generalized_dice_update(
            preds, target, self.num_classes, self.include_background, self.weight_type
        )
        self.score += _generalized_dice_compute(numerator, denominator, self.per_class)

    def compute(self):
        return self.score
    
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
            >>> from torchmetrics.audio import PerceptualEvaluationSpeechQuality
            >>> metric = PerceptualEvaluationSpeechQuality(8000, 'nb')
            >>> metric.update(torch.rand(8000), torch.rand(8000))
            >>> fig_, ax_ = metric.plot()
        .. plot::
            :scale: 75
            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.audio import PerceptualEvaluationSpeechQuality
            >>> metric = PerceptualEvaluationSpeechQuality(8000, 'nb')
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(torch.rand(8000), torch.rand(8000)))
            >>> fig_, ax_ = metric.plot(values)
        """
        return self._plot(val, ax)
