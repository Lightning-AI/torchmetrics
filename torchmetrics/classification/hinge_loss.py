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

from torchmetrics.functional.classification.hinge_loss import _hinge_loss_compute, _hinge_loss_update
from torchmetrics.metric import Metric


class HingeLoss(Metric):

    def __init__(
        self,
        squared: bool = False,
        multiclass_mode: Optional[str] = None,
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

        self.add_state("loss", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

        self.squared = squared
        self.multiclass_mode = multiclass_mode

    def update(self, preds: Tensor, target: Tensor):
        loss, total = _hinge_loss_update(preds, target, squared=self.squared, multiclass_mode=self.multiclass_mode)

        self.loss = loss + self.loss
        self.total = total + self.total

    def compute(self) -> Tensor:
        return _hinge_loss_compute(self.loss, self.total)
