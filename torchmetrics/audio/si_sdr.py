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

from torch import Tensor, tensor

from torchmetrics.functional.audio.si_sdr import si_sdr
from torchmetrics.metric import Metric


class SI_SDR(Metric):
    """Scale-invariant signal-to-distortion ratio (SI-SDR). The SI-SDR value is in general considered an overall
    measure of how good a source sound.

    Forward accepts

    - ``preds``: ``shape [...,time]``
    - ``target``: ``shape [...,time]``

    Args:
        zero_mean:
            if to zero mean target and preds or not
        compute_on_step:
            Forward only calls ``update()`` and returns None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step.
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When `None`, DDP
            will be used to perform the allgather.

    Raises:
        TypeError:
            if target and preds have a different shape

    Returns:
        average si-sdr value

    Example:
        >>> import torch
        >>> from torchmetrics import SI_SDR
        >>> target = torch.tensor([3.0, -0.5, 2.0, 7.0])
        >>> preds = torch.tensor([2.5, 0.0, 2.0, 8.0])
        >>> si_sdr = SI_SDR()
        >>> si_sdr_val = si_sdr(preds, target)
        >>> si_sdr_val
        tensor(18.4030)

    References:
        [1] Le Roux, Jonathan, et al. "SDR half-baked or well done." IEEE International Conference on Acoustics, Speech
        and Signal Processing (ICASSP) 2019.
    """

    sum_si_sdr: Tensor
    total: Tensor

    def __init__(
        self,
        zero_mean: bool = False,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.zero_mean = zero_mean

        self.add_state("sum_si_sdr", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        si_sdr_batch = si_sdr(preds=preds, target=target, zero_mean=self.zero_mean)

        self.sum_si_sdr += si_sdr_batch.sum()
        self.total += si_sdr_batch.numel()

    def compute(self) -> Tensor:
        """Computes average SI-SDR."""
        return self.sum_si_sdr / self.total

    @property
    def is_differentiable(self) -> bool:
        return True
