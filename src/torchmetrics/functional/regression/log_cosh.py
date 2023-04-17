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

from torchmetrics.functional.regression.utils import _check_data_shape_to_num_outputs
from torchmetrics.utilities.checks import _check_same_shape


def _unsqueeze_tensors(preds: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    if preds.ndim == 2:
        return preds, target
    return preds.unsqueeze(1), target.unsqueeze(1)


def _log_cosh_error_update(preds: Tensor, target: Tensor, num_outputs: int) -> Tuple[Tensor, Tensor]:
    """Update and returns variables required to compute LogCosh error.

    Check for same shape of input tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        num_outputs: Number of outputs in multioutput setting

    Return:
        Sum of LogCosh error over examples, and total number of examples
    """
    _check_same_shape(preds, target)
    _check_data_shape_to_num_outputs(preds, target, num_outputs)

    preds, target = _unsqueeze_tensors(preds, target)
    diff = preds - target
    sum_log_cosh_error = torch.log((torch.exp(diff) + torch.exp(-diff)) / 2).sum(0).squeeze()
    n_obs = torch.tensor(target.shape[0], device=preds.device)
    return sum_log_cosh_error, n_obs


def _log_cosh_error_compute(sum_log_cosh_error: Tensor, n_obs: Tensor) -> Tensor:
    """Compute Mean Squared Error.

    Args:
        sum_log_cosh_error: Sum of LogCosh errors over all observations
        n_obs: Number of predictions or observations
    """
    return (sum_log_cosh_error / n_obs).squeeze()


def log_cosh_error(preds: Tensor, target: Tensor) -> Tensor:
    r"""Compute the `LogCosh Error`_.

    .. math:: \text{LogCoshError} = \log\left(\frac{\exp(\hat{y} - y) + \exp(\hat{y - y})}{2}\right)

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a tensor of predictions.

    Args:
        preds: estimated labels with shape ``(batch_size,)`` or `(batch_size, num_outputs)``
        target: ground truth labels with shape ``(batch_size,)`` or `(batch_size, num_outputs)``

    Return:
        Tensor with LogCosh error

    Example (single output regression)::
        >>> from torchmetrics.functional.regression import log_cosh_error
        >>> preds = torch.tensor([3.0, 5.0, 2.5, 7.0])
        >>> target = torch.tensor([2.5, 5.0, 4.0, 8.0])
        >>> log_cosh_error(preds, target)
        tensor(0.3523)

    Example (multi output regression)::
        >>> from torchmetrics.functional.regression import log_cosh_error
        >>> preds = torch.tensor([[3.0, 5.0, 1.2], [-2.1, 2.5, 7.0]])
        >>> target = torch.tensor([[2.5, 5.0, 1.3], [0.3, 4.0, 8.0]])
        >>> log_cosh_error(preds, target)
        tensor([0.9176, 0.4277, 0.2194])
    """
    sum_log_cosh_error, n_obs = _log_cosh_error_update(
        preds, target, num_outputs=1 if preds.ndim == 1 else preds.shape[-1]
    )
    return _log_cosh_error_compute(sum_log_cosh_error, n_obs)
