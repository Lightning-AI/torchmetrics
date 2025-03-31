# Copyright The Lightning team.
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


def _crps_update(preds: Tensor, target: Tensor) -> Tuple[int, Tensor, Tensor]:
    """Compute intermediate CRPS values before aggregation.

    Args:
        preds: Tensor of shape (batch_size, ensemble_members)
        target: Tensor of shape (batch_size,)

    Returns:
        batch_size: int
        diff: Tensor (batch-wise absolute error term)
        ensemble_sum: Tensor (pairwise ensemble term)

    """
    # Only second dimension should deviate in shape (the ensemble members)
    _check_same_shape(preds[:, 0], target)

    batch_size, n_ensemble_members = preds.shape
    if n_ensemble_members < 2:
        raise ValueError(f"CRPS requires at least 2 ensemble members, but you provided {preds.shape}.")

    # sort forecasts
    preds = torch.sort(preds, dim=1)[0]

    # inflate observations:
    observation_inflated = target.unsqueeze(1).expand_as(preds)

    # Compute mean absolute difference between predictions and target
    diff = torch.sum(torch.abs(preds - observation_inflated), dim=1) / n_ensemble_members

    # Compute ensemble term using the reference implementation formula
    ensemble_diffs = torch.abs(preds.unsqueeze(2) - preds.unsqueeze(1))
    ensemble_sum = torch.sum(ensemble_diffs, dim=(1, 2)) / (2 * n_ensemble_members * n_ensemble_members)

    return batch_size, diff, ensemble_sum


def _crps_compute(batch_size: int, diff: Tensor, ensemble_sum: Tensor) -> Tensor:
    """Final CRPS computation."""
    return torch.mean(diff - ensemble_sum)  # Changed from sum to mean


def continuous_ranked_probability_score(preds: Tensor, target: Tensor) -> Tensor:
    r"""Computes continuous ranked probability score.

    .. math::
        CRPS(F, y) = \int_{-\infty}^{\infty} (F(x) - 1_{x \geq y})^2 dx

    where :math:`F` is the predicted cumulative distribution function and :math:`y` is the true target. The metric is
    usually used to evaluate probabilistic regression models, such as forecasting models. A lower CRPS indicates a
    better forecast, meaning that forecasted probabilities are closer to the true observed values. CRPS can also be
    seen as a generalization of the brier score for non binary classification problems.

    Args:
        preds: a 2d tensor of shape (batch_size, ensemble_members) with predictions. The second dimension represents
            the ensemble members.
        target: a 1d tensor of shape (batch_size) with the target values.

    Return:
        Tensor with CRPS

    Raises:
        ValueError:
            If the number of ensemble members is less than 2.
        ValueError:
            If the first dimension of preds and target do not match.

    Example::
        >>> from torchmetrics.functional.regression import continuous_ranked_probability_score
        >>> from torch import randn
        >>> preds = randn(10, 5)
        >>> target = randn(10)
        >>> continuous_ranked_probability_score(preds, target)
        tensor(0.7731)

    """
    batch_size, diff, ensemble_sum = _crps_update(preds, target)
    return _crps_compute(batch_size, diff, ensemble_sum)
