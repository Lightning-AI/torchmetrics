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
from typing import Any

from torch import Tensor, tensor

from torchmetrics.functional.audio.snr import scale_invariant_signal_noise_ratio, signal_noise_ratio
from torchmetrics.metric import Metric


class SignalNoiseRatio(Metric):
    r"""Signal-to-noise ratio (SNR_):

    .. math::
        \text{SNR} = \frac{P_{signal}}{P_{noise}}

    where  :math:`P` denotes the power of each signal. The SNR metric compares the level
    of the desired signal to the level of background noise. Therefore, a high value of
    SNR means that the audio is clear.

    Forward accepts

    - ``preds``: ``shape [..., time]``
    - ``target``: ``shape [..., time]``

    Args:
        zero_mean: if to zero mean target and preds or not

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        TypeError:
            if target and preds have a different shape

    Returns:
        average snr value

    Example:
        >>> import torch
        >>> from torchmetrics import SignalNoiseRatio
        >>> target = torch.tensor([3.0, -0.5, 2.0, 7.0])
        >>> preds = torch.tensor([2.5, 0.0, 2.0, 8.0])
        >>> snr = SignalNoiseRatio()
        >>> snr(preds, target)
        tensor(16.1805)

    References:
        [1] Le Roux, Jonathan, et al. "SDR half-baked or well done." IEEE International Conference on Acoustics, Speech
        and Signal Processing (ICASSP) 2019.

    """
    full_state_update: bool = False
    is_differentiable: bool = True
    higher_is_better: bool = True
    sum_snr: Tensor
    total: Tensor

    def __init__(
        self,
        zero_mean: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.zero_mean = zero_mean

        self.add_state("sum_snr", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        snr_batch = signal_noise_ratio(preds=preds, target=target, zero_mean=self.zero_mean)

        self.sum_snr += snr_batch.sum()
        self.total += snr_batch.numel()

    def compute(self) -> Tensor:
        """Computes average SNR."""
        return self.sum_snr / self.total


class ScaleInvariantSignalNoiseRatio(Metric):
    """Scale-invariant signal-to-noise ratio (SI-SNR).

    Forward accepts

    - ``preds``: ``shape [...,time]``
    - ``target``: ``shape [...,time]``

    Args:
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        TypeError:
            if target and preds have a different shape

    Returns:
        average si-snr value

    Example:
        >>> import torch
        >>> from torchmetrics import ScaleInvariantSignalNoiseRatio
        >>> target = torch.tensor([3.0, -0.5, 2.0, 7.0])
        >>> preds = torch.tensor([2.5, 0.0, 2.0, 8.0])
        >>> si_snr = ScaleInvariantSignalNoiseRatio()
        >>> si_snr(preds, target)
        tensor(15.0918)

    References:
        [1] Y. Luo and N. Mesgarani, "TaSNet: Time-Domain Audio Separation Network for Real-Time, Single-Channel Speech
        Separation," 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2018, pp.
        696-700, doi: 10.1109/ICASSP.2018.8462116.
    """

    is_differentiable = True
    sum_si_snr: Tensor
    total: Tensor
    higher_is_better = True

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.add_state("sum_si_snr", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        si_snr_batch = scale_invariant_signal_noise_ratio(preds=preds, target=target)

        self.sum_si_snr += si_snr_batch.sum()
        self.total += si_snr_batch.numel()

    def compute(self) -> Tensor:
        """Computes average SI-SNR."""
        return self.sum_si_snr / self.total
