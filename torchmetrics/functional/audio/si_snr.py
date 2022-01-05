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
from warnings import warn

from torch import Tensor

from torchmetrics.functional.audio.snr import scale_invariant_signal_noise_ratio


def si_snr(preds: Tensor, target: Tensor) -> Tensor:
    """Scale-invariant signal-to-noise ratio (SI-SNR).

    .. deprecated:: v0.7
        Use :func:`torchmetrics.functional.scale_invariant_signal_noise_ratio`. Will be removed in v0.8.

    Example:
        >>> import torch
        >>> si_snr(torch.tensor([2.5, 0.0, 2.0, 8.0]), torch.tensor([3.0, -0.5, 2.0, 7.0]))
        tensor(15.0918)
    """
    warn(
        "`si_snr` was renamed to `scale_invariant_signal_noise_ratio` in v0.7 and it will be removed in v0.8",
        DeprecationWarning,
    )
    return scale_invariant_signal_noise_ratio(preds, target)
