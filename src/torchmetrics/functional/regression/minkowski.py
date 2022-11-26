from typing import Optional

import torch
from torch import Tensor

from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.exceptions import TorchMetricsUserError


def _minkowski_distance_update(preds: Tensor, targets: Tensor, p: float) -> Tensor:
    _check_same_shape(preds, targets)

    if p < 0:
        raise TorchMetricsUserError("p value must be greater than 0")

    difference = torch.abs(preds - targets)
    mink_dist_sum = torch.sum(torch.pow(difference, p))

    return mink_dist_sum


def _minkowski_distance_compute(distance: Tensor, p: float) -> Tensor:
    return torch.pow(distance, 1.0 / p)


def minkowski_distance(preds: Tensor, targets: Tensor, p: float) -> Tensor:
    minkowski_dist_sum = _minkowski_distance_update(preds, targets, p)
    return _minkowski_distance_compute(minkowski_dist_sum, p)
