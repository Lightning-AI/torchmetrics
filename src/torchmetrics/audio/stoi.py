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

from torch import Tensor, tensor

from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _PYSTOI_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

__doctest_requires__ = {"ShortTimeObjectiveIntelligibility": ["pystoi"]}

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["ShortTimeObjectiveIntelligibility.plot"]


class ShortTimeObjectiveIntelligibility(Metric):
    r"""Calculate STOI (Short-Time Objective Intelligibility) metric for evaluating speech signals.

    Intelligibility measure which is highly correlated with the intelligibility of degraded speech signals, e.g., due
    to additive noise, single-/multi-channel noise reduction, binary masking and vocoded speech as in CI simulations.
    The STOI-measure is intrusive, i.e., a function of the clean and degraded speech signals. STOI may be a good
    alternative to the speech intelligibility index (SII) or the speech transmission index (STI), when you are
    interested in the effect of nonlinear processing to noisy speech, e.g., noise reduction, binary masking algorithms,
    on speech intelligibility. Description taken from  `Cees Taal's website`_ and for further defails see `STOI ref1`_
    and `STOI ref2`_.

    This metric is a wrapper for the `pystoi package`_. As the implementation backend implementation only supports
    calculations on CPU, all input will automatically be moved to CPU to perform the metric calculation before being
    moved back to the original device.

    As input to `forward` and `update` the metric accepts the following input

    - ``preds`` (:class:`~torch.Tensor`): float tensor with shape ``(...,time)``
    - ``target`` (:class:`~torch.Tensor`): float tensor with shape ``(...,time)``

    As output of `forward` and `compute` the metric returns the following output

    - ``stoi`` (:class:`~torch.Tensor`): float scalar tensor

    .. note:: using this metrics requires you to have ``pystoi`` install. Either install as ``pip install
        torchmetrics[audio]`` or ``pip install pystoi``.

    Args:
        fs: sampling frequency (Hz)
        extended: whether to use the extended STOI described in `STOI ref3`_.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ModuleNotFoundError:
            If ``pystoi`` package is not installed

    Example:
        >>> import torch
        >>> from torchmetrics.audio import ShortTimeObjectiveIntelligibility
        >>> g = torch.manual_seed(1)
        >>> preds = torch.randn(8000)
        >>> target = torch.randn(8000)
        >>> stoi = ShortTimeObjectiveIntelligibility(8000, False)
        >>> stoi(preds, target)
        tensor(-0.0100)
    """
    sum_stoi: Tensor
    total: Tensor
    full_state_update: bool = False
    is_differentiable: bool = False
    higher_is_better: bool = True
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def __init__(
        self,
        fs: int,
        extended: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not _PYSTOI_AVAILABLE:
            raise ModuleNotFoundError(
                "STOI metric requires that `pystoi` is installed."
                " Either install as `pip install torchmetrics[audio]` or `pip install pystoi`."
            )
        self.fs = fs
        self.extended = extended

        self.add_state("sum_stoi", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        stoi_batch = short_time_objective_intelligibility(preds, target, self.fs, self.extended, False).to(
            self.sum_stoi.device
        )

        self.sum_stoi += stoi_batch.sum()
        self.total += stoi_batch.numel()

    def compute(self) -> Tensor:
        """Compute metric."""
        return self.sum_stoi / self.total

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
            >>> from torchmetrics.audio import ShortTimeObjectiveIntelligibility
            >>> g = torch.manual_seed(1)
            >>> preds = torch.randn(8000)
            >>> target = torch.randn(8000)
            >>> metric = ShortTimeObjectiveIntelligibility(8000, False)
            >>> metric.update(preds, target)
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.audio import ShortTimeObjectiveIntelligibility
            >>> metric = ShortTimeObjectiveIntelligibility(8000, False)
            >>> g = torch.manual_seed(1)
            >>> preds = torch.randn(8000)
            >>> target = torch.randn(8000)
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(preds, target))
            >>> fig_, ax_ = metric.plot(values)
        """
        return self._plot(val, ax)
