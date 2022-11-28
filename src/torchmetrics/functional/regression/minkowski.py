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
from typing import Optional

import torch
from torch import Tensor

from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.exceptions import TorchMetricsUserError


def _minkowski_distance_update(preds: Tensor, targets: Tensor, p: float) -> Tensor:
    """Updates and returns variables required to compute Minkowski distance.

    Checks for same shape of input tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        p: Non-negative number acting as the exponent to the errors
    """
    _check_same_shape(preds, targets)

    if not isinstance(p, (float, int)) and p < 0:
        raise TorchMetricsUserError(f"Argument `p` must be a float or int greater than 0, but got {p}")

    difference = torch.abs(preds - targets)
    mink_dist_sum = torch.sum(torch.pow(difference, p))

    return mink_dist_sum


def _minkowski_distance_compute(distance: Tensor, p: float) -> Tensor:
    """Computes Minkowski Distance.

    Args:
        distance: Sum of the p-th powers of errors over all observations
        p: The non-negative numeric power the errors are to be raised to

    Example:
        >>> preds = torch.tensor([0., 1, 2, 3])
        >>> target = torch.tensor([0., 2, 3, 1])
        >>> distance_p_sum = _minkowski_distance_update(preds, target, 5)
        >>> _minkowski_distance_compute(distance_p_sum, 5)
        tensor(2.0244)
    """
    return torch.pow(distance, 1.0 / p)


def minkowski_distance(preds: Tensor, targets: Tensor, p: float) -> Tensor:
    """Computes the Minkowski distance.

    Args:
        preds: estimated labels of type Tensor
        target: ground truth labels of type Tensor
        p: non-negative numeric (int or float) exponent to which the difference
           between preds and target is to be raised

    Return:
        Tensor with the Minkowski distance
    Example:
        >>> from torchmetrics.functional import minkowski_distance
        >>> x = torch.tensor([1.0, 2.8, 3.5, 4.5])
        >>> y = torch.tensor([6.1, 2.11, 3.1, 5.6])
        >>> minkowski_distance(x, y)
        tensor(5.1220)
    """
    minkowski_dist_sum = _minkowski_distance_update(preds, targets, p)
    return _minkowski_distance_compute(minkowski_dist_sum, p)
