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
from warnings import warn

import torch
from torch import Tensor

from torchmetrics.functional.regression.mean_absolute_percentage_error import (
    _mean_absolute_percentage_error_compute,
    _mean_absolute_percentage_error_update,
)


def mean_relative_error(preds: Tensor, target: Tensor) -> Tensor:
    """
    Computes mean relative error

    Args:
        preds: estimated labels
        target: ground truth labels

    Return:
        Tensor with mean relative error

    Example:
        >>> from torchmetrics.functional import mean_relative_error
        >>> x = torch.tensor([0., 1, 2, 3])
        >>> y = torch.tensor([0., 1, 2, 2])
        >>> mean_relative_error(x, y)
        tensor(0.1250)

    .. deprecated::
        Use :func:`torchmetrics.functional.mean_absolute_percentage_error`. Will be removed in v0.5.0.

    """
    warn(
        "Function `mean_relative_error` was deprecated v0.4 and will be removed in v0.5."
        "Use `mean_absolute_percentage_error` instead.", DeprecationWarning
    )
    sum_rltv_error, n_obs = _mean_absolute_percentage_error_update(preds, target)
    return _mean_absolute_percentage_error_compute(sum_rltv_error, n_obs)
