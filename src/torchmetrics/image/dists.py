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
from typing import Literal, Optional, Sequence, Union

from torch import Tensor

from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["DeepImageStructureAndTextureSimilarity.plot"]


class DeepImageStructureAndTextureSimilarity(Metric):
    """Calculates Deep Image Structure and Texture Similarity (DISTS) score."""

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0

    def __init__(self, reduction: Optional[Literal["mean", "sum", "none"]] = "mean") -> None:
        super().__init__()

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update the metric state."""

    def compute(self) -> Tensor:
        """Computes the DISTS score."""

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
            >>> from torchmetrics.image.lpip import DeepImageStructureAndTextureSimilarity
            >>> metric = DeepImageStructureAndTextureSimilarity(net_type='squeeze')
            >>> metric.update(torch.rand(10, 3, 100, 100), torch.rand(10, 3, 100, 100))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.image.lpip import DeepImageStructureAndTextureSimilarity
            >>> metric = DeepImageStructureAndTextureSimilarity(net_type='squeeze')
            >>> values = [ ]
            >>> for _ in range(3):
            ...     values.append(metric(torch.rand(10, 3, 100, 100), torch.rand(10, 3, 100, 100)))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)
