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
from typing import Any, Optional, Callable

import torch
from torch import Tensor, tensor

from torchmetrics.functional.regression.crps import _crps_compute, _crps_update
from torchmetrics.metric import Metric


class CRPS(Metric):
    r"""
    CRPS metric calculates the Continuous Ranked Probability Score which corresponds to the
    that is mean squared error between a predicted cumulative density function (CDF) and the
    true underlying CDF:

    .. math::
        CRPS(F, x) = \int_{-\infty}^\infty (F(y) - \mathbb{1}(y - x))^2 dy

    where :math:`F` is the CDF function and :math:`x` is a observation. It should be noted
    the CRPS metric is closely related to the `Brier Score <https://en.wikipedia.org/wiki/Brier_score>`_.

    Forward accepts

    - ``preds`` (float tensor): ``(N,d)``
    - ``target`` (float tensor): ``(N,d)``

    Args:
        compute_on_step:
            Forward only calls ``update()`` and return ``None`` if this is set to ``False``.
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step.
        process_group:
            Specify the process group on which synchronization is called.
            default: ``None`` (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When ``None``, DDP
            will be used to perform the all gather.

    Example:
        >>> from torchmetrics import CosineSimilarity
        >>> target = torch.tensor([[0, 1], [1, 1]])
        >>> preds = torch.tensor([[0, 1], [0, 1]])
        >>> cosine_similarity = CosineSimilarity(reduction = 'mean')
        >>> cosine_similarity(preds, target)

    """

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn
        )

        self.add_state("batch_size", default=tensor(0), dist_reduce_fx="sum")
        self.add_state("diff", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("ensemble_sum_scale_factor", default=tensor(1.0), dist_reduce_fx="mean")
        self.add_state("ensemble_sum", default=tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        batch_size, diff, ensemble_sum_scale_factor, ensemble_sum = _crps_update(preds, target)
        self.batch_size += batch_size
        self.diff += diff
        self.ensemble_sum_scale_factor = ensemble_sum_scale_factor
        self.ensemble_sum += ensemble_sum

    def compute(self) -> Tensor:
        """
        Compute the continuous ranked probability score over state.
        """
        return _crps_compute(self.batch_size, self.diff, self.ensemble_sum_scale_factor, self.ensemble_sum)
