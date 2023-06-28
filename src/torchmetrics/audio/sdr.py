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

from torchmetrics.functional.audio.sdr import scale_invariant_signal_distortion_ratio, signal_distortion_ratio
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

__doctest_requires__ = {"SignalDistortionRatio": ["fast_bss_eval"]}

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["SignalDistortionRatio.plot", "ScaleInvariantSignalDistortionRatio.plot"]


class SignalDistortionRatio(Metric):
    r"""Calculate Signal to Distortion Ratio (SDR) metric.

    See `SDR ref1`_ and `SDR ref2`_ for details on the metric.

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``preds`` (:class:`~torch.Tensor`): float tensor with shape ``(...,time)``
    - ``target`` (:class:`~torch.Tensor`): float tensor with shape ``(...,time)``

    As output of `forward` and `compute` the metric returns the following output

    - ``sdr`` (:class:`~torch.Tensor`): float scalar tensor with average SDR value over samples

    .. note:
        The metric currently does not seem to work with Pytorch v1.11 and specific GPU hardware.

    Args:
        use_cg_iter:
            If provided, conjugate gradient descent is used to solve for the distortion
            filter coefficients instead of direct Gaussian elimination, which requires that
            ``fast-bss-eval`` is installed and pytorch version >= 1.8.
            This can speed up the computation of the metrics in case the filters
            are long. Using a value of 10 here has been shown to provide
            good accuracy in most cases and is sufficient when using this
            loss to train neural separation networks.
        filter_length: The length of the distortion filter allowed
        zero_mean:
            When set to True, the mean of all signals is subtracted prior to computation of the metrics
        load_diag:
            If provided, this small value is added to the diagonal coefficients of the system metrics when solving
            for the filter coefficients. This can help stabilize the metric in the case where some reference
            signals may sometimes be zero
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> import torch
        >>> from torchmetrics.audio import SignalDistortionRatio
        >>> g = torch.manual_seed(1)
        >>> preds = torch.randn(8000)
        >>> target = torch.randn(8000)
        >>> sdr = SignalDistortionRatio()
        >>> sdr(preds, target)
        tensor(-12.0589)
        >>> # use with pit
        >>> from torchmetrics.audio import PermutationInvariantTraining
        >>> from torchmetrics.functional.audio import signal_distortion_ratio
        >>> preds = torch.randn(4, 2, 8000)  # [batch, spk, time]
        >>> target = torch.randn(4, 2, 8000)
        >>> pit = PermutationInvariantTraining(signal_distortion_ratio,
        ...     mode="speaker-wise", eval_func="max")
        >>> pit(preds, target)
        tensor(-11.6051)
    """

    sum_sdr: Tensor
    total: Tensor
    full_state_update: bool = False
    is_differentiable: bool = True
    higher_is_better: bool = True
    plot_lower_bound: Optional[float] = None
    plot_upper_bound: Optional[float] = None

    def __init__(
        self,
        use_cg_iter: Optional[int] = None,
        filter_length: int = 512,
        zero_mean: bool = False,
        load_diag: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.use_cg_iter = use_cg_iter
        self.filter_length = filter_length
        self.zero_mean = zero_mean
        self.load_diag = load_diag

        self.add_state("sum_sdr", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        sdr_batch = signal_distortion_ratio(
            preds, target, self.use_cg_iter, self.filter_length, self.zero_mean, self.load_diag
        )

        self.sum_sdr += sdr_batch.sum()
        self.total += sdr_batch.numel()

    def compute(self) -> Tensor:
        """Compute metric."""
        return self.sum_sdr / self.total

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
            >>> from torchmetrics.audio import SignalDistortionRatio
            >>> metric = SignalDistortionRatio()
            >>> metric.update(torch.rand(8000), torch.rand(8000))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.audio import SignalDistortionRatio
            >>> metric = SignalDistortionRatio()
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(torch.rand(8000), torch.rand(8000)))
            >>> fig_, ax_ = metric.plot(values)
        """
        return self._plot(val, ax)


class ScaleInvariantSignalDistortionRatio(Metric):
    """`Scale-invariant signal-to-distortion ratio`_ (SI-SDR).

    The SI-SDR value is in general considered an overall measure of how good a source sound.

    As input to `forward` and `update` the metric accepts the following input

    - ``preds`` (:class:`~torch.Tensor`): float tensor with shape ``(...,time)``
    - ``target`` (: :class:`~torch.Tensor`): float tensor with shape ``(...,time)``

    As output of `forward` and `compute` the metric returns the following output

    - ``si_sdr`` (: :class:`~torch.Tensor`): float scalar tensor with average SI-SDR value over samples

    Args:
        zero_mean: if to zero mean target and preds or not
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        TypeError:
            if target and preds have a different shape

    Example:
        >>> from torch import tensor
        >>> from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
        >>> target = tensor([3.0, -0.5, 2.0, 7.0])
        >>> preds = tensor([2.5, 0.0, 2.0, 8.0])
        >>> si_sdr = ScaleInvariantSignalDistortionRatio()
        >>> si_sdr(preds, target)
        tensor(18.4030)
    """

    is_differentiable = True
    higher_is_better = True
    sum_si_sdr: Tensor
    total: Tensor
    plot_lower_bound: Optional[float] = None
    plot_upper_bound: Optional[float] = None

    def __init__(
        self,
        zero_mean: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.zero_mean = zero_mean

        self.add_state("sum_si_sdr", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        si_sdr_batch = scale_invariant_signal_distortion_ratio(preds=preds, target=target, zero_mean=self.zero_mean)

        self.sum_si_sdr += si_sdr_batch.sum()
        self.total += si_sdr_batch.numel()

    def compute(self) -> Tensor:
        """Compute metric."""
        return self.sum_si_sdr / self.total

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
            >>> from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
            >>> target = torch.randn(5)
            >>> preds = torch.randn(5)
            >>> metric = ScaleInvariantSignalDistortionRatio()
            >>> metric.update(preds, target)
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
            >>> target = torch.randn(5)
            >>> preds = torch.randn(5)
            >>> metric = ScaleInvariantSignalDistortionRatio()
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(preds, target))
            >>> fig_, ax_ = metric.plot(values)
        """
        return self._plot(val, ax)
