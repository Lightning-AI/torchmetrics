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
from functools import partial
from typing import Any, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, tensor
from typing_extensions import Literal

from torchmetrics.functional.image.psnr import _psnr_compute, _psnr_update
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["PeakSignalNoiseRatio.plot"]


class PeakSignalNoiseRatio(Metric):
    r"""`Compute Peak Signal-to-Noise Ratio`_ (PSNR).

    .. math:: \text{PSNR}(I, J) = 10 * \log_{10} \left(\frac{\max(I)^2}{\text{MSE}(I, J)}\right)

    Where :math:`\text{MSE}` denotes the `mean-squared-error`_ function.

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``preds`` (:class:`~torch.Tensor`): Predictions from model of shape ``(N,C,H,W)``
    - ``target`` (:class:`~torch.Tensor`): Ground truth values of shape ``(N,C,H,W)``

    As output of `forward` and `compute` the metric returns the following output

    - ``psnr`` (:class:`~torch.Tensor`): if ``reduction!='none'`` returns float scalar tensor with average PSNR value
      over sample else returns tensor of shape ``(N,)`` with PSNR values per sample

    Args:
        data_range:
            the range of the data. If None, it is determined from the data (max - min). If a tuple is provided then
            the range is calculated as the difference and input is clamped between the values.
            The ``data_range`` must be given when ``dim`` is not None.
        base: a base of a logarithm to use.
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

        dim:
            Dimensions to reduce PSNR scores over, provided as either an integer or a list of integers. Default is
            None meaning scores will be reduced across all dimensions and all batches.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``dim`` is not ``None`` and ``data_range`` is not given.

    Example:
        >>> from torchmetrics.image import PeakSignalNoiseRatio
        >>> psnr = PeakSignalNoiseRatio()
        >>> preds = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        >>> target = torch.tensor([[3.0, 2.0], [1.0, 0.0]])
        >>> psnr(preds, target)
        tensor(2.5527)
    """
    is_differentiable: bool = True
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0

    min_target: Tensor
    max_target: Tensor

    def __init__(
        self,
        data_range: Optional[Union[float, Tuple[float, float]]] = None,
        base: float = 10.0,
        reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
        dim: Optional[Union[int, Tuple[int, ...]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if dim is None and reduction != "elementwise_mean":
            rank_zero_warn(f"The `reduction={reduction}` will not have any effect when `dim` is None.")

        if dim is None:
            self.add_state("sum_squared_error", default=tensor(0.0), dist_reduce_fx="sum")
            self.add_state("total", default=tensor(0), dist_reduce_fx="sum")
        else:
            self.add_state("sum_squared_error", default=[], dist_reduce_fx="cat")
            self.add_state("total", default=[], dist_reduce_fx="cat")

        self.clamping_fn = None
        if data_range is None:
            if dim is not None:
                # Maybe we could use `torch.amax(target, dim=dim) - torch.amin(target, dim=dim)` in PyTorch 1.7 to
                # calculate `data_range` in the future.
                raise ValueError("The `data_range` must be given when `dim` is not None.")

            self.data_range = None
            self.add_state("min_target", default=tensor(0.0), dist_reduce_fx=torch.min)
            self.add_state("max_target", default=tensor(0.0), dist_reduce_fx=torch.max)
        elif isinstance(data_range, tuple):
            self.add_state("data_range", default=tensor(data_range[1] - data_range[0]), dist_reduce_fx="mean")
            self.clamping_fn = partial(torch.clamp, min=data_range[0], max=data_range[1])
        else:
            self.add_state("data_range", default=tensor(float(data_range)), dist_reduce_fx="mean")
        self.base = base
        self.reduction = reduction
        self.dim = tuple(dim) if isinstance(dim, Sequence) else dim

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        if self.clamping_fn is not None:
            preds = self.clamping_fn(preds)
            target = self.clamping_fn(target)

        sum_squared_error, n_obs = _psnr_update(preds, target, dim=self.dim)
        if self.dim is None:
            if self.data_range is None:
                # keep track of min and max target values
                self.min_target = torch.minimum(target.min(), self.min_target)
                self.max_target = torch.maximum(target.max(), self.max_target)

            self.sum_squared_error += sum_squared_error
            self.total += n_obs
        else:
            self.sum_squared_error.append(sum_squared_error)
            self.total.append(n_obs)

    def compute(self) -> Tensor:
        """Compute peak signal-to-noise ratio over state."""
        data_range = self.data_range if self.data_range is not None else self.max_target - self.min_target

        if self.dim is None:
            sum_squared_error = self.sum_squared_error
            total = self.total
        else:
            sum_squared_error = torch.cat([values.flatten() for values in self.sum_squared_error])
            total = torch.cat([values.flatten() for values in self.total])
        return _psnr_compute(sum_squared_error, total, data_range, base=self.base, reduction=self.reduction)

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
            >>> from torchmetrics.image import PeakSignalNoiseRatio
            >>> metric = PeakSignalNoiseRatio()
            >>> preds = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
            >>> target = torch.tensor([[3.0, 2.0], [1.0, 0.0]])
            >>> metric.update(preds, target)
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.image import PeakSignalNoiseRatio
            >>> metric = PeakSignalNoiseRatio()
            >>> preds = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
            >>> target = torch.tensor([[3.0, 2.0], [1.0, 0.0]])
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(preds, target))
            >>> fig_, ax_ = metric.plot(values)
        """
        return self._plot(val, ax)
