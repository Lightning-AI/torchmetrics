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

from torch import Tensor, tensor

from torchmetrics.functional.audio.nisqa import non_intrusive_speech_quality_assessment
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import (
    _LIBROSA_AVAILABLE,
    _MATPLOTLIB_AVAILABLE,
    _REQUESTS_AVAILABLE,
)
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

__doctest_requires__ = {"NonIntrusiveSpeechQualityAssessment": ["librosa", "requests"]}

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["NonIntrusiveSpeechQualityAssessment.plot"]


class NonIntrusiveSpeechQualityAssessment(Metric):
    """`Non-Intrusive Speech Quality Assessment`_ (NISQA v2.0) [1], [2].

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``preds`` (:class:`~torch.Tensor`): float tensor with shape ``(...,time)``

    As output of ``forward`` and ``compute`` the metric returns the following output

    - ``nisqa`` (:class:`~torch.Tensor`): float tensor reduced across the batch with shape ``(5,)`` corresponding to
      overall MOS, noisiness, discontinuity, coloration and loudness in that order

    .. hint::
        Using this metric requires you to have ``librosa`` and ``requests`` installed. Install as
        ``pip install librosa requests``.

    .. caution::
        The ``forward`` and ``compute`` methods in this class return values reduced across the batch. To obtain
        values for each sample, you may use the functional counterpart
        :func:`~torchmetrics.functional.audio.nisqa.non_intrusive_speech_quality_assessment`.

    Args:
        fs: sampling frequency of input

    Raises:
        ModuleNotFoundError:
            If ``librosa`` or ``requests`` are not installed

    Example:
        >>> import torch
        >>> from torchmetrics.audio import NonIntrusiveSpeechQualityAssessment
        >>> _ = torch.manual_seed(42)
        >>> preds = torch.randn(16000)
        >>> nisqa = NonIntrusiveSpeechQualityAssessment(16000)
        >>> nisqa(preds)
        tensor([1.0433, 1.9545, 2.6087, 1.3460, 1.7117])

    References:
        - [1] G. Mittag and S. Möller, "Non-intrusive speech quality assessment for super-wideband speech communication
          networks", in Proc. ICASSP, 2019.
        - [2] G. Mittag, B. Naderi, A. Chehadi and S. Möller, "NISQA: A deep CNN-self-attention model for
          multidimensional speech quality prediction with crowdsourced datasets", in Proc. INTERSPEECH, 2021.

    """

    sum_nisqa: Tensor
    total: Tensor
    full_state_update: bool = False
    is_differentiable: bool = False
    higher_is_better: bool = True
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 5.0

    def __init__(self, fs: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not _LIBROSA_AVAILABLE or not _REQUESTS_AVAILABLE:
            raise ModuleNotFoundError(
                "NISQA metric requires that librosa and requests are installed. "
                "Install as `pip install librosa requests`."
            )
        if not isinstance(fs, int) or fs <= 0:
            raise ValueError(f"Argument `fs` expected to be a positive integer, but got {fs}")
        self.fs = fs

        self.add_state("sum_nisqa", default=tensor([0.0, 0.0, 0.0, 0.0, 0.0]), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor) -> None:
        """Update state with predictions."""
        nisqa_batch = non_intrusive_speech_quality_assessment(
            preds,
            self.fs,
        ).to(self.sum_nisqa.device)

        nisqa_batch = nisqa_batch.reshape(-1, 5)
        self.sum_nisqa += nisqa_batch.sum(dim=0)
        self.total += nisqa_batch.shape[0]

    def compute(self) -> Tensor:
        """Compute metric."""
        return self.sum_nisqa / self.total

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
            >>> from torchmetrics.audio import NonIntrusiveSpeechQualityAssessment
            >>> metric = NonIntrusiveSpeechQualityAssessment(16000)
            >>> metric.update(torch.randn(16000))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.audio import NonIntrusiveSpeechQualityAssessment
            >>> metric = NonIntrusiveSpeechQualityAssessment(16000)
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(torch.randn(16000)))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)
