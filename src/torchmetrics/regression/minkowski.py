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
    r"""Computes `Minkowski Distance`

    .. math:: d_{\text{Minkowski}} = \\sum_{i}^N (| y_i - \\hat{y_i} |^p)^\frac{1}{p}

    where
        :math:`y` is a tensor of target values,
        :math:`\\hat{y}` is a tensor of predictions,
        :math: `\\p` is a non-negative integer or floating-point number

    Args:
        p: A non-negative number acting as the exponent in the calculation
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> import torch
        >>> from torchmetrics import MinkowskiDistance
        >>> target = torch.tensor([1.0, 2.8, 3.5, 4.5])
        >>> preds = torch.tensor([6.1, 2.11, 3.1, 5.6])
        >>> minkowski_distance = MinkowskiDistance(3)
        >>> minkowski_distance(preds, target)
        tensor(5.1220)
    """

    is_differentiable: Optional[bool] = True
    higher_is_better: Optional[bool] = False
    full_state_update: Optional[bool] = False
    minkowski_dist_sum: Tensor

    def __init__(self, p: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not isinstance(p, float) and p < 0:
            raise TorchMetricsUserError(f"Argument `p` must be a float greater than 0, but got {p}")
        self.p = p
        self.add_state("minkowski_dist_sum", default=tensor(0.0), dist_sync_fn="sum")

    def update(self, preds: Tensor, targets: Tensor) -> None:
        minkowski_dist_sum = _minkowski_distance_update(preds, targets, self.p)

        self.minkowski_dist_sum += minkowski_dist_sum

    def compute(self) -> Tensor:
        return _minkowski_distance_compute(self.minkowski_dist_sum, self.p)
