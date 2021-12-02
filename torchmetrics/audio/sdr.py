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

from torchmetrics.functional.audio.sdr import sdr
from torchmetrics.metric import Metric


class SDR(Metric):
    r"""Signal to Distortion Ratio (SDR) [1,2,3]

    Forward accepts

    - ``preds``: shape ``[..., time]``
    - ``target``: shape ``[..., time]``

    Args:
        use_cg_iter:
            If provided, an iterative method is used to solve for the distortion
            filter coefficients instead of direct Gaussian elimination.
            This can speed up the computation of the metrics in case the filters
            are long. Using a value of 10 here has been shown to provide
            good accuracy in most cases and is sufficient when using this
            loss to train neural separation networks.
        filter_length:
            The length of the distortion filter allowed
        zero_mean:
            When set to True, the mean of all signals is subtracted prior to computation of the metrics
        load_diag:
            If provided, this small value is added to the diagonal coefficients of
            the system metrics when solving for the filter coefficients.
            This can help stabilize the metric in the case where some of the reference
            signals may sometimes be zero
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
        ValueError:
            If ``fast-bss-eval`` package is not installed

    Example:
        >>> from torchmetrics.audio import SDR
        >>> import torch
        >>> g = torch.manual_seed(1)
        >>> preds = torch.randn(8000)
        >>> target = torch.randn(8000)
        >>> sdr = SDR()
        >>> sdr(preds, target)
        tensor(-12.0589)
        >>> # use with pit
        >>> from torchmetrics.audio import PIT
        >>> from torchmetrics.functional.audio import sdr
        >>> preds = torch.randn(4, 2, 8000)  # [batch, spk, time]
        >>> target = torch.randn(4, 2, 8000)
        >>> pit = PIT(sdr, 'max')
        >>> pit(preds, target)
        tensor(-11.6051)

    .. note::
       1. when pytorch<1.8.0, numpy will be used to calculate this metric, which causes ``sdr`` non-differentiable

       2. using this metrics requires you to have ``fast-bss-eval`` install. Either install as ``pip install
       torchmetrics[audio]`` or ``pip install fast-bss-eval``

       3. preds and target need to have the same dtype, otherwise target will be converted to preds' dtype


    References:
        [1] Vincent, E., Gribonval, R., & Fevotte, C. (2006). Performance measurement in blind audio source separation.
         IEEE Transactions on Audio, Speech and Language Processing, 14(4), 1462–1469.
        [2] Scheibler, R. (2021). SDR -- Medium Rare with Fast Computations.
        [3] https://github.com/fakufaku/fast_bss_eval
    """

    sum_sdr: Tensor
    total: Tensor
    is_differentiable = True
    higher_is_better = True

    def __init__(
        self,
        use_cg_iter: Optional[int] = None,
        filter_length: int = 512,
        zero_mean: bool = False,
        load_diag: Optional[float] = None,
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

        self.use_cg_iter = use_cg_iter
        self.filter_length = filter_length
        self.zero_mean = zero_mean
        self.load_diag = load_diag

        self.add_state("sum_sdr", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        sdr_batch = sdr(preds, target, self.use_cg_iter, self.filter_length, self.zero_mean, self.load_diag)

        self.sum_sdr += sdr_batch.sum()
        self.total += sdr_batch.numel()

    def compute(self) -> Tensor:
        """Computes average SDR."""
        return self.sum_sdr / self.total
