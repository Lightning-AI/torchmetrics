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

from typing import Tuple

import torch
from torch import Tensor

from torchmetrics.utilities.checks import _check_same_shape


def _crps_update(preds: Tensor, target: Tensor) -> Tuple[int, Tensor, float, float]:
    # Only second dimension should deviate in shape (the ensemble members)
    _check_same_shape(preds[:, 0], target)

    batch_size = target.shape[0]
    n_ensemble_members = preds.shape[1]

    # inflate observations:
    observation_inflated = torch.ones_like(preds)
    for i in range(n_ensemble_members):
        observation_inflated[:, i] = target

    diff = (1 / n_ensemble_members) * torch.sum(torch.abs(preds - observation_inflated))

    ensemble_sum_scale_factor = (1 / (n_ensemble_members * (n_ensemble_members - 1))) if n_ensemble_members > 1 else 1.0
    ensemble_sum_scale_factor = torch.tensor(ensemble_sum_scale_factor, device=preds.device)

    ensemble_sum = 0
    for i in range(n_ensemble_members):
        for j in range(i, n_ensemble_members):
            ensemble_sum += torch.sum(torch.abs(preds[:, i] - preds[:, j]))

    return batch_size, diff, ensemble_sum_scale_factor, ensemble_sum


def _crps_compute(batch_size: int, diff: Tensor, ensemble_sum_scale_factor: float, ensemble_sum: Tensor) -> Tensor:
    return (1 / batch_size) * (diff - (ensemble_sum_scale_factor * ensemble_sum))


def crps(preds: Tensor, target: Tensor) -> Tensor:
    """Computes continuous ranked probability score.

    Args:
        preds: estimated labels. Second dimension is the number of ensemble members.
        target: ground truth labels. Has to be the same shape as preds except for the ensemble member dimension, which
                should be missing for the targets.

    Return:
        Tensor with CRPS
    """
    batch_size, diff, ensemble_sum_scale_factor, ensemble_sum = _crps_update(preds, target)
    return _crps_compute(batch_size, diff, ensemble_sum_scale_factor, ensemble_sum)
