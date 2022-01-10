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

from torchmetrics.functional.audio.stoi import stoi
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _PYSTOI_AVAILABLE


class STOI(Metric):
    r"""STOI (Short Term Objective Intelligibility, see [2,3]), a wrapper for the pystoi package [1].
    Note that input will be moved to `cpu` to perform the metric calculation.

    Intelligibility measure which is highly correlated with the intelligibility of degraded speech signals, e.g., due
    to additive noise, single/multi-channel noise reduction, binary masking and vocoded speech as in CI simulations.
    The STOI-measure is intrusive, i.e., a function of the clean and degraded speech signals. STOI may be a good
    alternative to the speech intelligibility index (SII) or the speech transmission index (STI), when you are
    interested in the effect of nonlinear processing to noisy speech, e.g., noise reduction, binary masking algorithms,
    on speech intelligibility. Description taken from [Cees Taal's website](http://www.ceestaal.nl/code/).

    .. note:: using this metrics requires you to have ``pystoi`` install. Either install as ``pip install
        torchmetrics[audio]`` or ``pip install pystoi``

    Forward accepts

    - ``preds``: ``shape [...,time]``
    - ``target``: ``shape [...,time]``

    Args:
        fs:
            sampling frequency (Hz)
        extended:
            whether to use the extended STOI described in [4]
        compute_on_step:
            Forward only calls ``update()`` and returns None if this is set to False.
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step.
        process_group:
            Specify the process group on which synchronization is called.
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When `None`, DDP
            will be used to perform the allgather.

    Returns:
        average STOI value

    Raises:
        ModuleNotFoundError:
            If ``pystoi`` package is not installed

    Example:
        >>> from torchmetrics.audio.stoi import STOI
        >>> import torch
        >>> g = torch.manual_seed(1)
        >>> preds = torch.randn(8000)
        >>> target = torch.randn(8000)
        >>> stoi = STOI(8000, False)
        >>> stoi(preds, target)
        tensor(-0.0100)

    References:
        [1] https://github.com/mpariente/pystoi

        [2] C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'A Short-Time Objective Intelligibility Measure for
        Time-Frequency Weighted Noisy Speech', ICASSP 2010, Texas, Dallas.

        [3] C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'An Algorithm for Intelligibility Prediction of
        Time-Frequency Weighted Noisy Speech', IEEE Transactions on Audio, Speech, and Language Processing, 2011.

        [4] J. Jensen and C. H. Taal, 'An Algorithm for Predicting the Intelligibility of Speech Masked by Modulated
        Noise Maskers', IEEE Transactions on Audio, Speech and Language Processing, 2016.

    """
    sum_stoi: Tensor
    total: Tensor
    is_differentiable = False
    higher_is_better = True

    def __init__(
        self,
        fs: int,
        extended: bool = False,
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
        if not _PYSTOI_AVAILABLE:
            raise ModuleNotFoundError(
                "STOI metric requires that `pystoi` is installed."
                " Either install as `pip install torchmetrics[audio]` or `pip install pystoi`."
            )
        self.fs = fs
        self.extended = extended

        self.add_state("sum_stoi", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        stoi_batch = stoi(preds, target, self.fs, self.extended, False).to(self.sum_stoi.device)

        self.sum_stoi += stoi_batch.sum()
        self.total += stoi_batch.numel()

    def compute(self) -> Tensor:
        """Computes average STOI."""
        return self.sum_stoi / self.total
