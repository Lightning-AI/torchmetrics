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

from torchmetrics.functional.audio.si_snr import si_snr
from torchmetrics.metric import Metric


class SI_SNR(Metric):
    """Scale-invariant signal-to-noise ratio (SI-SNR).

    Forward accepts

    - ``preds``: ``shape [...,time]``
    - ``target``: ``shape [...,time]``

    Args:
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
        average si-snr value

    Example:
        >>> import torch
        >>> from torchmetrics import SI_SNR
        >>> target = torch.tensor([3.0, -0.5, 2.0, 7.0])
        >>> preds = torch.tensor([2.5, 0.0, 2.0, 8.0])
        >>> si_snr = SI_SNR()
        >>> si_snr_val = si_snr(preds, target)
        >>> si_snr_val
        tensor(15.0918)

    References:
        [1] Y. Luo and N. Mesgarani, "TaSNet: Time-Domain Audio Separation Network for Real-Time, Single-Channel Speech
        Separation," 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2018, pp.
        696-700, doi: 10.1109/ICASSP.2018.8462116.
    """

    sum_si_snr: Tensor
    total: Tensor

    def __init__(
        self,
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

        self.add_state("sum_si_snr", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        si_snr_batch = si_snr(preds=preds, target=target)

        self.sum_si_snr += si_snr_batch.sum()
        self.total += si_snr_batch.numel()

    def compute(self) -> Tensor:
        """Computes average SI-SNR."""
        return self.sum_si_snr / self.total

    @property
    def is_differentiable(self) -> bool:
        return True
