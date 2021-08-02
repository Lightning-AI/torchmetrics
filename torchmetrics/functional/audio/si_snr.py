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
from torch import Tensor

from torchmetrics.functional.audio.si_sdr import si_sdr


def si_snr(preds: Tensor, target: Tensor) -> Tensor:
    """Scale-invariant signal-to-noise ratio (SI-SNR).

    Args:
        preds:
            shape ``[...,time]``
        target:
            shape ``[...,time]``

    Returns:
        si-snr value of shape [...]

    Example:
        >>> import torch
        >>> from torchmetrics.functional.audio import si_snr
        >>> target = torch.tensor([3.0, -0.5, 2.0, 7.0])
        >>> preds = torch.tensor([2.5, 0.0, 2.0, 8.0])
        >>> si_snr_val = si_snr(preds, target)
        >>> si_snr_val
        tensor(15.0918)

    References:
        [1] Y. Luo and N. Mesgarani, "TaSNet: Time-Domain Audio Separation Network for Real-Time, Single-Channel Speech
        Separation," 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2018, pp.
        696-700, doi: 10.1109/ICASSP.2018.8462116.
    """

    return si_sdr(target=target, preds=preds, zero_mean=True)
