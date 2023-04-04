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
from typing import Any, Dict, List, Optional, Sequence, Union

from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.image.sam import _sam_compute, _sam_update
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["SpectralAngleMapper.plot"]


class SpectralAngleMapper(Metric):
    """`Spectral Angle Mapper`_ determines the spectral similarity between image spectra and reference spectra.

    It works by calculating the angle between the spectra, where small angles between indicate high similarity and
    high angles indicate low similarity.

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``preds`` (:class:`~torch.Tensor`): Predictions from model of shape ``(N,C,H,W)``
    - ``target`` (:class:`~torch.Tensor`): Ground truth values of shape ``(N,C,H,W)``

    As output of `forward` and `compute` the metric returns the following output

    - ``sam`` (:class:`~torch.Tensor`): if ``reduction!='none'`` returns float scalar tensor with average SAM value
      over sample else returns tensor of shape ``(N,)`` with SAM values per sample

    Args:
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Return:
        Tensor with SpectralAngleMapper score

    Example:
        >>> import torch
        >>> from torchmetrics.image import SpectralAngleMapper
        >>> preds = torch.rand([16, 3, 16, 16], generator=torch.manual_seed(42))
        >>> target = torch.rand([16, 3, 16, 16], generator=torch.manual_seed(123))
        >>> sam = SpectralAngleMapper()
        >>> sam(preds, target)
        tensor(0.5943)
    """

    higher_is_better: bool = False
    is_differentiable: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
        self,
        reduction: Literal["elementwise_mean", "sum", "none"] = "elementwise_mean",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        rank_zero_warn(
            "Metric `SpectralAngleMapper` will save all targets and predictions in the buffer."
            " For large datasets, this may lead to a large memory footprint."
        )

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.reduction = reduction

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        preds, target = _sam_update(preds, target)
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> Tensor:
        """Compute spectra over state."""
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        return _sam_compute(preds, target, self.reduction)

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

            >>> # Example plotting single value
            >>> import torch
            >>> from torchmetrics.image import SpectralAngleMapper
            >>> preds = torch.rand([16, 3, 16, 16], generator=torch.manual_seed(42))
            >>> target = torch.rand([16, 3, 16, 16], generator=torch.manual_seed(123))
            >>> metric = SpectralAngleMapper()
            >>> metric.update(preds, target)
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.image import SpectralAngleMapper
            >>> preds = torch.rand([16, 3, 16, 16], generator=torch.manual_seed(42))
            >>> target = torch.rand([16, 3, 16, 16], generator=torch.manual_seed(123))
            >>> metric = SpectralAngleMapper()
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(preds, target))
            >>> fig_, ax_ = metric.plot(values)
        """
        return self._plot(val, ax)
