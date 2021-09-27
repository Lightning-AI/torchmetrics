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
from torchmetrics.utilities.imports import _PESQ_AVAILABLE


class PESQ(Metric):
    """PESQ (Perceptual Evaluation of Speech Quality)

    This is a wrapper for the pesq package [1]. . Note that input will be moved to `cpu`
    to perform the metric calculation.

    .. note:: using this metrics requires you to have ``pesq`` install. Either install as ``pip install
        torchmetrics[audio]`` or ``pip install pesq``

    Forward accepts

    - ``preds``: ``shape [...,time]``
    - ``target``: ``shape [...,time]``

    Args:
        fs:
            sampling frequency, should be 16000 or 8000 (Hz)
        mode:
            'wb' (wide-band) or 'nb' (narrow-band)
        keep_same_device:
            whether to move the pesq value to the device of preds
        compute_on_step:
            Forward only calls ``update()`` and return ``None`` if this is set to ``False``.
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step
        process_group:
            Specify the process group on which synchronization is called.
            default: ``None`` (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When ``None``, DDP
            will be used to perform the allgather

    Raises:
        ValueError:
            If ``peqs`` package is not installed
        ValueError:
            If ``fs`` is not either  ``8000`` or ``16000``
        ValueError:
            If ``mode`` is not either ``"wb"`` or ``"nb"``

    Example:
        >>> from torchmetrics.audio import PESQ
        >>> import torch
        >>> g = torch.manual_seed(1)
        >>> preds = torch.randn(8000)
        >>> target = torch.randn(8000)
        >>> nb_pesq = PESQ(8000, 'nb')
        >>> nb_pesq(preds, target)
        tensor(2.2076)
        >>> wb_pesq = PESQ(16000, 'wb')
        >>> wb_pesq(preds, target)
        tensor(1.7359)

    References:
        [1] https://github.com/ludlows/python-pesq
    """

    sum_pesq: Tensor
    total: Tensor
    is_differentiable = False
    higher_is_better = True

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
        if not _PESQ_AVAILABLE:
            raise ValueError(
                "PESQ metric requires that pesq is installed."
                "Either install as `pip install torchmetrics[audio]` or `pip install pesq`"
            )
        if fs not in (8000, 16000):
            raise ValueError(f"Expected argument `fs` to either be 8000 or 16000 but got {fs}")
        self.fs = fs
        if mode not in ("wb", "nb"):
            raise ValueError(f"Expected argument `mode` to either be 'wb' or 'nb' but got {mode}")
        self.mode = mode

        self.add_state("sum_pesq", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        pesq_batch = pesq(preds, target, self.fs, self.mode, False).to(self.sum_pesq.device)

        self.sum_pesq += pesq_batch.sum()
        self.total += pesq_batch.numel()

    def compute(self) -> Tensor:
        """Computes average PESQ."""
        return self.sum_pesq / self.total
