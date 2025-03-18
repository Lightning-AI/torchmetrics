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
from torch import Tensor, tensor

from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import (
    _LIBROSA_AVAILABLE,
    _MATPLOTLIB_AVAILABLE,
    _ONNXRUNTIME_AVAILABLE,
    _REQUESTS_AVAILABLE,
)
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

__doctest_requires__ = {"DeepNoiseSuppressionMeanOpinionScore": ["requests", "librosa", "onnxruntime"]}

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["DeepNoiseSuppressionMeanOpinionScore.plot"]


class DeepNoiseSuppressionMeanOpinionScore(Metric):
    """Calculate `Deep Noise Suppression performance evaluation based on Mean Opinion Score`_ (DNSMOS).

    Human subjective evaluation is the ”gold standard” to evaluate speech quality optimized for human perception.
    Perceptual objective metrics serve as a proxy for subjective scores. The conventional and widely used metrics
    require a reference clean speech signal, which is unavailable in real recordings. The no-reference approaches
    correlate poorly with human ratings and are not widely adopted in the research community. One of the biggest
    use cases of these perceptual objective metrics is to evaluate noise suppression algorithms. DNSMOS generalizes
    well in challenging test conditions with a high correlation to human ratings in stack ranking noise suppression
    methods. More details can be found in `DNSMOS paper <https://arxiv.org/abs/2010.15258>`_ and
    `DNSMOS P.835 paper <https://arxiv.org/abs/2110.01763>`_.


    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``preds`` (:class:`~torch.Tensor`): float tensor with shape ``(...,time)``

    As output of ``forward`` and ``compute`` the metric returns the following output

    - ``dnsmos`` (:class:`~torch.Tensor`): float tensor of DNSMOS values reduced across the batch
        with shape ``(...,4)`` indicating [p808_mos, mos_sig, mos_bak, mos_ovr] in the last dim.

    .. hint::
        Using this metric requires you to have ``librosa``, ``onnxruntime`` and ``requests`` installed.
        Install as ``pip install torchmetrics['audio']`` or alternatively `pip install librosa onnxruntime-gpu requests`
        (if you do not have GPU enabled machine install `onnxruntime` instead of `onnxruntime-gpu`)

    .. caution::
        The ``forward`` and ``compute`` methods in this class return a reduced DNSMOS value
        for a batch. To obtain the DNSMOS value for each sample, you may use the functional counterpart in
        :func:`~torchmetrics.functional.audio.dnsmos.deep_noise_suppression_mean_opinion_score`.

    Args:
        fs: sampling frequency
        personalized: whether interfering speaker is penalized
        device: the device used for calculating DNSMOS, can be cpu or cuda:n, where n is the index of gpu.
            If None is given, then the device of input is used.
        num_threads: number of threads to use for onnxruntime CPU inference.
        cache_session: whether to cache the onnx session. By default this is true, meaning that repeated calls to this
            method is faster than if this was set to False, the consequence is that the session will be cached in
            memory until the process is terminated.

    Raises:
        ModuleNotFoundError:
            If ``librosa``, ``onnxruntime`` or ``requests`` packages are not installed

    Example:
        >>> from torch import randn
        >>> from torchmetrics.audio import DeepNoiseSuppressionMeanOpinionScore
        >>> preds = randn(8000)
        >>> dnsmos = DeepNoiseSuppressionMeanOpinionScore(8000, False)
        >>> dnsmos(preds)
        tensor([2.2..., 2.0..., 1.1..., 1.2...], dtype=torch.float64)

    """

    sum_dnsmos: Tensor
    total: Tensor
    full_state_update: bool = False
    is_differentiable: bool = False
    higher_is_better: bool = True
    plot_lower_bound: float = 0
    plot_upper_bound: float = 5

    def __init__(
        self,
        fs: int,
        personalized: bool,
        device: Optional[str] = None,
        num_threads: Optional[int] = None,
        cache_sessions: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not _LIBROSA_AVAILABLE or not _ONNXRUNTIME_AVAILABLE or not _REQUESTS_AVAILABLE:
            raise ModuleNotFoundError(
                "DNSMOS metric requires that librosa, onnxruntime and requests are installed."
                " Install as `pip install librosa onnxruntime-gpu requests`."
            )
        if fs <= 0 or not isinstance(fs, int):
            raise ValueError("Argument `fs` must be a positive integer.")
        self.fs = fs

        if not isinstance(personalized, bool):
            raise ValueError("Argument `personalized` must be a boolean.")
        self.personalized = personalized

        self.cal_device = device
        self.num_threads = num_threads
        self.cache_sessions = cache_sessions

        self.add_state("sum_dnsmos", default=tensor([0, 0, 0, 0], dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor) -> None:
        """Update state with predictions."""
        metric_batch = deep_noise_suppression_mean_opinion_score(
            preds=preds,
            fs=self.fs,
            personalized=self.personalized,
            device=self.cal_device,
            num_threads=self.num_threads,
            cache_session=self.cache_sessions,
        ).to(self.sum_dnsmos.device)

        self.sum_dnsmos += metric_batch.reshape(-1, 4).sum(dim=0)
        self.total += metric_batch.reshape(-1, 4).shape[0]

    def compute(self) -> Tensor:
        """Compute metric."""
        return self.sum_dnsmos / self.total

    def plot(self, val: Union[Tensor, Sequence[Tensor], None] = None, ax: Optional[_AX_TYPE] = None) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling ``metric.forward`` or ``metric.compute`` or a list of these
                results. If no value is provided, will automatically call ``metric.compute`` and plot that result.
            ax: A matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If ``matplotlib`` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> import torch
            >>> from torchmetrics.audio import DeepNoiseSuppressionMeanOpinionScore
            >>> metric = DeepNoiseSuppressionMeanOpinionScore(8000, False)
            >>> metric.update(torch.rand(8000))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.audio import DeepNoiseSuppressionMeanOpinionScore
            >>> metric = DeepNoiseSuppressionMeanOpinionScore(8000, False)
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(torch.rand(8000)))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)
