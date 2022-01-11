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
from typing import Any, Callable, Optional

from deprecate import deprecated, void
from torch import Tensor

from torchmetrics.audio.sdr import ScaleInvariantSignalDistortionRatio


class SI_SDR(ScaleInvariantSignalDistortionRatio):
    """Scale-invariant signal-to-distortion ratio (SI-SDR).

    .. deprecated:: v0.7
        Use :class:`torchmetrics.ScaleInvariantSignalDistortionRatio`. Will be removed in v0.8.

    Example:
        >>> import torch
        >>> si_sdr = SI_SDR()
        >>> si_sdr(torch.tensor([2.5, 0.0, 2.0, 8.0]), torch.tensor([3.0, -0.5, 2.0, 7.0]))
        tensor(18.4030)
    """

    @deprecated(target=ScaleInvariantSignalDistortionRatio, deprecated_in="0.7", remove_in="0.8")
    def __init__(
        self,
        zero_mean: bool = False,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> None:
        void(zero_mean, compute_on_step, dist_sync_on_step, process_group, dist_sync_fn)
