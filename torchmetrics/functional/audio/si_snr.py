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
from .si_sdr import si_sdr


def si_snr(target, estimate, EPS=1e-8):
    """ scale-invariant signal-to-noise ratio (SI-SNR)

    Args:
        target (Tensor): shape [..., time]
        estimate (Tensor): shape [..., time]
        EPS (float, optional): a small value for numerical stability. Defaults to 1e-8.

    Raises:
        TypeError: if target and estimate have a different shape

    Returns:
        Tensor: si-snr value has a shape of [...]

    Example:
        >>> from torchmetrics.functional.audio import si_snr
        >>> target = torch.tensor([3.0, -0.5, 2.0, 7.0])
        >>> estimate = torch.tensor([2.5, 0.0, 2.0, 8.0])
        >>> si_snr_val = si_snr(target,estimate)
        >>> si_snr_val
        tensor(15.0918)

    References:
        [1] Y. Luo and N. Mesgarani, "TaSNet: Time-Domain Audio Separation Network for Real-Time, Single-Channel Speech
         Separation," 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2018, pp.
         696-700, doi: 10.1109/ICASSP.2018.8462116.
    """

    return si_sdr(target=target, estimate=estimate, zero_mean=True, EPS=EPS)
