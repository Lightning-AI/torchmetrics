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
from deprecate import deprecated, void
from torch import Tensor

from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio


@deprecated(target=scale_invariant_signal_distortion_ratio, deprecated_in="0.7", remove_in="0.8")
def si_sdr(preds: Tensor, target: Tensor, zero_mean: bool = False) -> Tensor:
    """Calculates Scale-invariant signal-to-distortion ratio (SI-SDR) metric.

    .. deprecated:: v0.7
        Use :func:`torchmetrics.functional.scale_invariant_signal_distortion_ratio`. Will be removed in v0.8.

    Example:
        >>> si_sdr(torch.tensor([2.5, 0.0, 2.0, 8.0]), torch.tensor([3.0, -0.5, 2.0, 7.0]))
        tensor(18.4030)
    """
    return void(preds, target, zero_mean)
