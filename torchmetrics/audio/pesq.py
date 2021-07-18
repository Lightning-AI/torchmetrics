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

from torchmetrics.functional.audio.pesq import pesq
from torchmetrics.metric import Metric


class PESQ(Metric):
    """
    PESQ (Perceptual Evaluation of Speech Quality)

    This is a wrapper for the pesq package [1].

    Forward accepts

    - ``preds``: ``shape [...,time]``
    - ``target``: ``shape [...,time]``

    Args:
        preds:
            shape ``[...,time]``
        target:
            shape ``[...,time]``
        fs:
            sampling frequency, should be 16000 or 8000 (Hz)
        mode:
            'wb' (wide-band) or 'nb' (narrow-band)
        keep_same_device:
            whether to move the pesq value to the device of preds

    Example:
        >>> from torchmetrics.functional.audio import pesq
        >>> import torch
        >>> preds = torch.randn(8000)
        >>> target = torch.randn(8000)
        >>> nb_pesq_val = pesq(preds, target, 8000, 'nb')
        >>> wb_pesq_val = pesq(preds, target, 16000, 'wb')

    References:
        [1] https://github.com/ludlows/python-pesq
    """
    sum_pesq: Tensor
    total: Tensor

    def __init__(
        self,
        fs: int,
        mode: str,
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
        self.fs = fs
        self.mode = mode

        self.add_state("sum_pesq", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        pesq_batch = pesq(preds, target, self.fs, self.mode, False).to(self.sum_pesq.device)

        self.sum_pesq += pesq_batch.sum()
        self.total += pesq_batch.numel()

    def compute(self) -> Tensor:
        """
        Computes average SI-SDR.
        """
        return self.sum_pesq / self.total

    @property
    def is_differentiable(self) -> bool:
        return False
