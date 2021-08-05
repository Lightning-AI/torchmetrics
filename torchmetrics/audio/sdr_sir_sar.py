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
from typing import Any, Callable, Optional, Tuple

from torch import Tensor, tensor

from torchmetrics.functional.audio.sdr_sir_sar import sdr_sir_sar
from torchmetrics.metric import Metric


class SDR_SIR_SAR(Metric):
    r"""SDR_SIR_SAR evaluates the average SDR_SIR_SAR metric of preds and target. A wrapper for
    the mir_eval.separation.bss_eval_sources function.

    Forward accepts

    - ``preds``: ``shape [..., spk, time]``
    - ``target``: ``shape [..., spk, time]``

    Args:
        compute_permutation:
            whether to compute the metrics permutation invariantly. By default, it is False for we can use PIT to
            compute the permutation in a better way and in the sense of any metrics.
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

    Returns:
        average SDR, SIR, SAR values

    Example:
        >>> from torchmetrics.audio import SDR_SIR_SAR
        >>> import torch
        >>> preds = torch.randn(8000)
        >>> target = torch.randn(8000)
        >>> sdr_sir_sar = SDR_SIR_SAR()
        >>> sdr_val, sir_val, sar_val = sdr_sir_sar(preds, target)
    """

    sum_sdr: Tensor
    sum_sir: Tensor
    sum_sar: Tensor
    total: Tensor

    def __init__(
        self,
        compute_permutation: bool = False,
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

        self.compute_permutation = compute_permutation

        self.add_state("sum_sdr", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_sir", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_sar", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        sdr_batch, sir_batch, sar_batch = sdr_sir_sar(preds, target, self.compute_permutation, False)

        self.sum_sdr += sdr_batch.sum().to(self.sum_sdr.device)
        self.sum_sir += sir_batch.sum().to(self.sum_sdr.device)
        self.sum_sar += sar_batch.sum().to(self.sum_sdr.device)
        self.total += sdr_batch.numel()

    def compute(self) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Computes average SDR, SIR, SAR.
        """
        return self.sum_sdr / self.total, self.sum_sir / self.total, self.sum_sar / self.total

    @property
    def is_differentiable(self) -> bool:
        return False


class SDR(SDR_SIR_SAR):

    def __init__(
        self,
        compute_permutation: bool = False,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> None:
        super().__init__(
            compute_permutation=compute_permutation,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

    def compute(self) -> Tensor:
        """
        Computes average SDR.
        """
        return self.sum_sdr / self.total


class SIR(SDR_SIR_SAR):

    def __init__(
        self,
        compute_permutation: bool = False,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> None:
        super().__init__(
            compute_permutation=compute_permutation,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

    def compute(self) -> Tensor:
        """
        Computes average SIR.
        """
        return self.sum_sir / self.total


class SAR(SDR_SIR_SAR):

    def __init__(
        self,
        compute_permutation: bool = False,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> None:
        super().__init__(
            compute_permutation=compute_permutation,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

    def compute(self) -> Tensor:
        """
        Computes average SAR.
        """
        return self.sum_sar / self.total
