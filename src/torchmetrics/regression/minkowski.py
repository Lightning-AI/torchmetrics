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

from typing import Any, Optional

import torch
from torch import Tensor, tensor

from torchmetrics.functional.regression.minkowski import _minkowski_distance_compute, _minkowski_distance_update
from torchmetrics.metric import Metric


class MinkowskiDistance(Metric):
    is_differentiable: Optional[bool] = True
    minkowski_dist_sum: Tensor

    def __init__(self, p: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.add_state("minkowski_dist_sum", default=tensor(0.0))
        self.p = p

    def update(self, preds: Tensor, targets: Tensor) -> None:
        minkowski_dist_sum = _minkowski_distance_update(preds, targets, self.p)

        self.minkowski_dist_sum += minkowski_dist_sum

    def compute(self) -> Tensor:
        return _minkowski_distance_compute(self.minkowski_dist_sum, self.p)
