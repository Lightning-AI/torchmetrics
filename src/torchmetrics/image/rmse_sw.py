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

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from torchmetrics.functional.image.rmse_sw import _rmse_sw_compute, _rmse_sw_update
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["RootMeanSquaredErrorUsingSlidingWindow.plot"]


class RootMeanSquaredErrorUsingSlidingWindow(Metric):
    """Computes Root Mean Squared Error (RMSE) using sliding window.

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``preds`` (:class:`~torch.Tensor`): Predictions from model of shape ``(N,C,H,W)``
    - ``target`` (:class:`~torch.Tensor`): Ground truth values of shape ``(N,C,H,W)``

    As output of `forward` and `compute` the metric returns the following output

    - ``rmse_sw`` (:class:`~torch.Tensor`): returns float scalar tensor with average RMSE-SW value over sample

    Args:
        window_size: Sliding window used for rmse calculation
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from torchmetrics.image import RootMeanSquaredErrorUsingSlidingWindow
        >>> g = torch.manual_seed(22)
        >>> preds = torch.rand(4, 3, 16, 16)
        >>> target = torch.rand(4, 3, 16, 16)
        >>> rmse_sw = RootMeanSquaredErrorUsingSlidingWindow()
        >>> rmse_sw(preds, target)
        tensor(0.3999)

    Raises:
        ValueError: If ``window_size`` is not a positive integer.
    """

    higher_is_better: bool = False
    is_differentiable: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0

    rmse_val_sum: Tensor
    rmse_map: Optional[Tensor] = None
    total_images: Tensor

    def __init__(
        self,
        window_size: int = 8,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(**kwargs)
        if not isinstance(window_size, int) or isinstance(window_size, int) and window_size < 1:
            raise ValueError("Argument `window_size` is expected to be a positive integer.")
        self.window_size = window_size

        self.add_state("rmse_val_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_images", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        if self.rmse_map is None:
            _img_shape = target.shape[1:]  # channels, width, height
            self.rmse_map = torch.zeros(_img_shape, dtype=target.dtype, device=target.device)

        self.rmse_val_sum, self.rmse_map, self.total_images = _rmse_sw_update(
            preds, target, self.window_size, self.rmse_val_sum, self.rmse_map, self.total_images
        )

    def compute(self) -> Optional[Tensor]:
        """Compute Root Mean Squared Error (using sliding window) and potentially return RMSE map."""
        assert self.rmse_map is not None  # noqa: S101  # needed for mypy
        rmse, _ = _rmse_sw_compute(self.rmse_val_sum, self.rmse_map, self.total_images)
        return rmse

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
            >>> from torchmetrics.image import RootMeanSquaredErrorUsingSlidingWindow
            >>> metric = RootMeanSquaredErrorUsingSlidingWindow()
            >>> metric.update(torch.rand(4, 3, 16, 16), torch.rand(4, 3, 16, 16))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.image import RootMeanSquaredErrorUsingSlidingWindow
            >>> metric = RootMeanSquaredErrorUsingSlidingWindow()
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(torch.rand(4, 3, 16, 16), torch.rand(4, 3, 16, 16)))
            >>> fig_, ax_ = metric.plot(values)
        """
        return self._plot(val, ax)
