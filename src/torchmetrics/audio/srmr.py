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

from torchmetrics.functional.audio.srmr import (
    _srmr_arg_validate,
    speech_reverberation_modulation_energy_ratio,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import (
    _GAMMATONE_AVAILABEL,
    _MATPLOTLIB_AVAILABLE,
    _TORCHAUDIO_AVAILABEL,
    _TORCHAUDIO_GREATER_EQUAL_0_10,
)
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not all([_GAMMATONE_AVAILABEL, _TORCHAUDIO_AVAILABEL, _TORCHAUDIO_GREATER_EQUAL_0_10]):
    __doctest_skip__ = ["SpeechReverberationModulationEnergyRatio", "SpeechReverberationModulationEnergyRatio.plot"]
elif not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["SpeechReverberationModulationEnergyRatio.plot"]


class SpeechReverberationModulationEnergyRatio(Metric):
    """Calculate `Speech-to-Reverberation Modulation Energy Ratio`_ (SRMR).

    SRMR is a non-intrusive metric for speech quality and intelligibility based on
    a modulation spectral representation of the speech signal.
    This code is translated from `SRMRToolbox`_ and `SRMRpy`_.

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``preds`` (:class:`~torch.Tensor`): float tensor with shape ``(...,time)``

    As output of `forward` and `compute` the metric returns the following output

    - ``srmr`` (:class:`~torch.Tensor`): float scaler tensor

    .. note:: using this metrics requires you to have ``gammatone`` and ``torchaudio`` installed.
        Either install as ``pip install torchmetrics[audio]`` or ``pip install torchaudio``
        and ``pip install git+https://github.com/detly/gammatone``.

    .. note::
        This implementation is experimental, and might not be consistent with the matlab
        implementation `SRMRToolbox`_, especially the fast implementation.
        The slow versions, a) fast=False, norm=False, max_cf=128, b) fast=False, norm=True, max_cf=30, have
        a relatively small inconsistence.

    Args:
        fs: the sampling rate
        n_cochlear_filters: Number of filters in the acoustic filterbank
        low_freq: determines the frequency cutoff for the corresponding gammatone filterbank.
        min_cf: Center frequency in Hz of the first modulation filter.
        max_cf: Center frequency in Hz of the last modulation filter. If None is given,
            then 30 Hz will be used for `norm==False`, otherwise 128 Hz will be used.
        norm: Use modulation spectrum energy normalization
        fast: Use the faster version based on the gammatonegram.
            Note: this argument is inherited from `SRMRpy`_. As the translated code is based to pytorch,
            setting `fast=True` may slow down the speed for calculating this metric on GPU.

    Raises:
        ModuleNotFoundError:
            If ``gammatone`` or ``torchaudio`` package is not installed

    Example:
        >>> import torch
        >>> from torchmetrics.audio import SpeechReverberationModulationEnergyRatio
        >>> g = torch.manual_seed(1)
        >>> preds = torch.randn(8000)
        >>> srmr = SpeechReverberationModulationEnergyRatio(8000)
        >>> srmr(preds)
        tensor(0.3354)
    """

    msum: Tensor
    total: Tensor
    full_state_update: bool = False
    is_differentiable: bool = True
    higher_is_better: bool = True
    plot_lower_bound: Optional[float] = None
    plot_upper_bound: Optional[float] = None

    def __init__(
        self,
        fs: int,
        n_cochlear_filters: int = 23,
        low_freq: float = 125,
        min_cf: float = 4,
        max_cf: Optional[float] = None,
        norm: bool = False,
        fast: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not _TORCHAUDIO_AVAILABEL or not _TORCHAUDIO_GREATER_EQUAL_0_10 or not _GAMMATONE_AVAILABEL:
            raise ModuleNotFoundError(
                "speech_reverberation_modulation_energy_ratio requires you to have `gammatone` and"
                " `torchaudio>=0.10` installed. Either install as ``pip install torchmetrics[audio]`` or "
                "``pip install torchaudio>=0.10`` and ``pip install git+https://github.com/detly/gammatone``"
            )
        _srmr_arg_validate(
            fs=fs,
            n_cochlear_filters=n_cochlear_filters,
            low_freq=low_freq,
            min_cf=min_cf,
            max_cf=max_cf,
            norm=norm,
            fast=fast,
        )

        self.fs = fs
        self.n_cochlear_filters = n_cochlear_filters
        self.low_freq = low_freq
        self.min_cf = min_cf
        self.max_cf = max_cf
        self.norm = norm
        self.fast = fast

        self.add_state("msum", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor) -> None:
        """Update state with predictions."""
        metric_val_batch = speech_reverberation_modulation_energy_ratio(
            preds, self.fs, self.n_cochlear_filters, self.low_freq, self.min_cf, self.max_cf, self.norm, self.fast
        ).to(self.msum.device)

        self.msum += metric_val_batch.sum()
        self.total += metric_val_batch.numel()

    def compute(self) -> Tensor:
        """Compute metric."""
        return self.msum / self.total

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
            >>> from torchmetrics.audio import SpeechReverberationModulationEnergyRatio
            >>> metric = SpeechReverberationModulationEnergyRatio(8000)
            >>> metric.update(torch.rand(8000))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.audio import SpeechReverberationModulationEnergyRatio
            >>> metric = SpeechReverberationModulationEnergyRatio(8000)
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(torch.rand(8000)))
            >>> fig_, ax_ = metric.plot(values)
        """
        return self._plot(val, ax)
