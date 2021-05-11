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
from typing import Any, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, tensor

from torchmetrics.functional.regression.crps import _crps_compute, _crps_update
from torchmetrics.metric import Metric


class CRPS(Metric):
    r"""

    """

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )

        self.add_state("batch_size", default=1)
        self.add_state("diff", default=tensor(0))
        self.add_state("ensemble_sum_scale_factor", default=1.)
        self.add_state("ensemble_sum", default=0.)

    def update(self, preds: Tensor, target: Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        self.batch_size, self.diff, self.ensemble_sum_scale_factor, self.ensemble_sum = _crps_update(preds, target)

    def compute(self):
        """
        Compute the continuous ranked probability score over state.
        """
        return _crps_compute(self.batch_size, self.diff, self.ensemble_sum_scale_factor, self.ensemble_sum)
