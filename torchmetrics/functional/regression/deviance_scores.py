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
from typing import Optional, Tuple

import torch
from torch import Tensor

from torchmetrics.utilities.checks import _check_same_shape


def _deviance_score_update(
    preds: Tensor,
    targets: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Updates and returns variables required to compute Deviance Score for the given power. Checks for same shape
    of input tensors.

    Args:
        preds: Predicted tensor
        targets: Ground truth tensor
    """
    _check_same_shape(preds, targets)
    return preds, targets


def _deviance_score_compute(preds: Tensor, targets: Tensor, power: Optional[int] = 0) -> Tensor:
    """Computes Cosine Similarity.

    Args:
        preds: Predicted tensor
        targets: Ground truth tensor
        power:
            - power = 0 : Normal distribution, output corresponds to mean_squared_error. y_true and y_pred can be any
            real numbers.
            - power = 1 : Poisson distribution. Requires: y_true >= 0 and y_pred > 0.
            - power = 2 : Gamma distribution. Requires: y_true > 0 and y_pred > 0.

    Example:
        >>> targets = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> preds = torch.tensor([4.0, 3.0, 2.0, 1.0])
        >>> preds, targets = _deviance_score_update(preds, targets)
        >>> _deviance_score_compute(preds, targets, power=0)
        tensor(5.)
    """

    if power < 1 and power > 0:
        raise ValueError(f"Deviance Score is not defined for power={power}.")

    if power == 0:
        deviance_score = torch.pow(targets - preds, exponent=2)
    elif power == 1:
        # Poisson distribution
        if torch.any(preds <= 0) or torch.any(targets < 0):
            raise ValueError(f"For power={power}, 'preds' has to be strictly positive and targets cannot be negative.")

        deviance_score = 2 * (targets * torch.log(targets / preds) + preds - targets)
    elif power == 2:
        # Gamma distribution
        if torch.any(preds <= 0) or torch.any(targets <= 0):
            raise ValueError(f"For power={power}, both 'preds' and 'targets' have to be strictly positive.")

        deviance_score = 2 * (torch.log(preds / targets) + (targets / preds) - 1)
    else:
        term_1 = torch.pow(torch.max(targets, torch.zeros(targets.shape)), 2 - power) / ((1 - power) * (2 - power))
        term_2 = targets * torch.pow(preds, 1 - power) / (1 - power)
        term_3 = torch.pow(preds, 2 - power) / (2 - power)
        deviance_score = 2 * (term_1 - term_2 + term_3)

    return torch.mean(deviance_score)


def deviance_score(preds: Tensor, targets: Tensor, power: Optional[int] = 0) -> Tensor:
    r"""
    Computes the `Deviance Score <https://en.wikipedia.org/wiki/Tweedie_distribution#The_Tweedie_deviance>`_ between
    targets and predictions:

    .. math::
        deviance\_score(\hat{y},y) =
        \begin{cases}
        (\hat{y} - y)^2, & \text{for }power=0\\
        2 * (y * log(\frac{y}{\hat{y}}) + \hat{y} - y),  & \text{for }power=1\\
        2 * (log(\frac{\hat{y}}{y}) + \frac{y}{\hat{y}} - 1),  & \text{for }power=2\\
        2 * (\frac{(max(y,0))^{2}}{(1 - power)(2 - power)} - \frac{y(\hat{y})^{1 - power}}{1 - power} + \frac{(\hat{y})
            ^{2 - power}}{2 - power}), & \text{otherwise}
        \end{cases}

    where :math:`y` is a tensor of targets values, and :math:`\hat{y}` is a tensor of predictions.

    Args:
        preds: Predicted tensor with shape ``(N,d)``
        targets: Ground truth tensor with shape ``(N,d)``
        power:
            - power = 0 : Normal distribution. (Requires: y_true and y_pred can be any real numbers.)
            - power = 1 : Poisson distribution. (Requires: y_true >= 0 and y_pred > 0.)
            - power = 2 : Gamma distribution. (Requires: y_true > 0 and y_pred > 0.)

    Example:
        >>> from torchmetrics.functional.regression import deviance_score
        >>> targets = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> preds = torch.tensor([4.0, 3.0, 2.0, 1.0])
        >>> deviance_score(preds, targets, power=0)
        tensor(5.)

    """
    preds, targets = _deviance_score_update(preds, targets)
    return _deviance_score_compute(preds, targets, power=power)
