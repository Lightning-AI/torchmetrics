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

from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _PESQ_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

__doctest_requires__ = {"PerceptualEvaluationSpeechQuality": ["pesq"]}

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["PerceptualEvaluationSpeechQuality.plot"]


class PerceptualEvaluationSpeechQuality(Metric):
    """Calculate `Perceptual Evaluation of Speech Quality`_ (PESQ).

    It's a recognized industry standard for audio quality that takes into considerations characteristics such as:
    audio sharpness, call volume, background noise, clipping, audio interference ect. PESQ returns a score between
    -0.5 and 4.5 with the higher scores indicating a better quality.

    This metric is a wrapper for the `pesq package`_. Note that input will be moved to ``cpu`` to perform the metric
    calculation.

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``preds`` (:class:`~torch.Tensor`): float tensor with shape ``(...,time)``
    - ``target`` (:class:`~torch.Tensor`): float tensor with shape ``(...,time)``

    As output of `forward` and `compute` the metric returns the following output

    - ``pesq`` (:class:`~torch.Tensor`): float tensor with shape ``(...,)`` of PESQ value per sample

    .. note:: using this metrics requires you to have ``pesq`` install. Either install as ``pip install
        torchmetrics[audio]`` or ``pip install pesq``. ``pesq`` will compile with your currently
        installed version of numpy, meaning that if you upgrade numpy at some point in the future you will
        most likely have to reinstall ``pesq``.

    Args:
        fs: sampling frequency, should be 16000 or 8000 (Hz)
        mode: ``'wb'`` (wide-band) or ``'nb'`` (narrow-band)
        keep_same_device: whether to move the pesq value to the device of preds
        n_processes: integer specifiying the number of processes to run in parallel for the metric calculation.
            Only applies to batches of data and if ``multiprocessing`` package is installed.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ModuleNotFoundError:
            If ``pesq`` package is not installed
        ValueError:
            If ``fs`` is not either  ``8000`` or ``16000``
        ValueError:
            If ``mode`` is not either ``"wb"`` or ``"nb"``

    Example:
        >>> import torch
        >>> from torchmetrics.audio import PerceptualEvaluationSpeechQuality
        >>> g = torch.manual_seed(1)
        >>> preds = torch.randn(8000)
        >>> target = torch.randn(8000)
        >>> nb_pesq = PerceptualEvaluationSpeechQuality(8000, 'nb')
        >>> nb_pesq(preds, target)
        tensor(2.2076)
        >>> wb_pesq = PerceptualEvaluationSpeechQuality(16000, 'wb')
        >>> wb_pesq(preds, target)
        tensor(1.7359)
    """

    sum_pesq: Tensor
    total: Tensor
    full_state_update: bool = False
    is_differentiable: bool = False
    higher_is_better: bool = True
    plot_lower_bound: float = -0.5
    plot_upper_bound: float = 4.5

    def __init__(
        self,
        fs: int,
        mode: str,
        n_processes: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not _PESQ_AVAILABLE:
            raise ModuleNotFoundError(
                "PerceptualEvaluationSpeechQuality metric requires that `pesq` is installed."
                " Either install as `pip install torchmetrics[audio]` or `pip install pesq`."
            )
        if fs not in (8000, 16000):
            raise ValueError(f"Expected argument `fs` to either be 8000 or 16000 but got {fs}")
        self.fs = fs
        if mode not in ("wb", "nb"):
            raise ValueError(f"Expected argument `mode` to either be 'wb' or 'nb' but got {mode}")
        self.mode = mode
        if not isinstance(n_processes, int) and n_processes <= 0:
            raise ValueError(f"Expected argument `n_processes` to be an int larger than 0 but got {n_processes}")
        self.n_processes = n_processes

        self.add_state("sum_pesq", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        pesq_batch = perceptual_evaluation_speech_quality(
            preds, target, self.fs, self.mode, False, self.n_processes
        ).to(self.sum_pesq.device)

        self.sum_pesq += pesq_batch.sum()
        self.total += pesq_batch.numel()

    def compute(self) -> Tensor:
        """Compute metric."""
        return self.sum_pesq / self.total

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
