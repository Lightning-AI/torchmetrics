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

import torch
from torch import Tensor

from torchmetrics.functional.audio.sdr import scale_invariant_signal_distortion_ratio
from torchmetrics.utilities.checks import _check_same_shape


def signal_noise_ratio(preds: Tensor, target: Tensor, zero_mean: bool = False) -> Tensor:
    r"""Signal-to-noise ratio (SNR_):

    .. math::
        \text{SNR} = \frac{P_{signal}}{P_{noise}}

    where  :math:`P` denotes the power of each signal. The SNR metric compares the level
    of the desired signal to the level of background noise. Therefore, a high value of
    SNR means that the audio is clear.

    Args:
        preds: shape ``[...,time]``
        target: shape ``[...,time]``
        zero_mean: if to zero mean target and preds or not

    Returns:
        snr value of shape [...]

    Example:
        >>> from torchmetrics.functional.audio import signal_noise_ratio
        >>> target = torch.tensor([3.0, -0.5, 2.0, 7.0])
        >>> preds = torch.tensor([2.5, 0.0, 2.0, 8.0])
        >>> signal_noise_ratio(preds, target)
        tensor(16.1805)

    References:
        [1] Le Roux, Jonathan, et al. "SDR half-baked or well done." IEEE International Conference on Acoustics, Speech
        and Signal Processing (ICASSP) 2019.

    """
    _check_same_shape(preds, target)
    eps = torch.finfo(preds.dtype).eps

    if zero_mean:
        target = target - torch.mean(target, dim=-1, keepdim=True)
        preds = preds - torch.mean(preds, dim=-1, keepdim=True)

    noise = target - preds

    snr_value = (torch.sum(target**2, dim=-1) + eps) / (torch.sum(noise**2, dim=-1) + eps)
    snr_value = 10 * torch.log10(snr_value)

    return snr_value


def scale_invariant_signal_noise_ratio(preds: Tensor, target: Tensor) -> Tensor:
    """Scale-invariant signal-to-noise ratio (SI-SNR).

    Args:
        preds: shape ``[...,time]``
        target: shape ``[...,time]``

    Returns:
        si-snr value of shape [...]

    Example:
        >>> import torch
        >>> from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio
        >>> target = torch.tensor([3.0, -0.5, 2.0, 7.0])
        >>> preds = torch.tensor([2.5, 0.0, 2.0, 8.0])
        >>> scale_invariant_signal_noise_ratio(preds, target)
        tensor(15.0918)

    References:
        [1] Y. Luo and N. Mesgarani, "TaSNet: Time-Domain Audio Separation Network for Real-Time, Single-Channel Speech
        Separation," 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2018, pp.
        696-700, doi: 10.1109/ICASSP.2018.8462116.
    """
    return scale_invariant_signal_distortion_ratio(preds=preds, target=target, zero_mean=True)
