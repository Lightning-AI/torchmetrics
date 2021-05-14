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

import torch
from torch import Tensor, tensor

from torchmetrics.functional.regression.mean_absolute_percentage_error import (
    _mean_absolute_percentage_error_compute,
    _mean_absolute_percentage_error_update,
)
from torchmetrics.metric import Metric


class MeanAbsolutePercentageError(Metric):
    def __init__(
        self,
        eps: float = 1.17e-07,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("sum_abs_per_error", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0.0), dist_reduce_fx="sum")
        self.eps = torch.tensor(eps)

    def update(self, preds: Tensor, target: Tensor) -> None:

        sum_abs_per_error, num_obs = _mean_absolute_percentage_error_update(preds, target, eps=self.eps)

        self.sum_abs_per_error += sum_abs_per_error
        self.total += num_obs

    def compute(self):
        return _mean_absolute_percentage_error_compute(self.sum_abs_per_error, self.total)

    @property
    def is_differentiable(self):
        return False
