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
import math
import warnings
from collections import Counter
from typing import Optional, Union

import scipy.stats as ss
import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.nominal.utils import _handle_nan_in_data


def _thiels_u_input_validation(nan_strategy: str, nan_replace_value: Optional[Union[int, float]]) -> None:
    if nan_strategy not in ["replace", "drop"]:
        raise ValueError(
            f"Argument `nan_strategy` is expected to be one of `['replace', 'drop']`, but got {nan_strategy}"
        )
    if nan_strategy == "replace" and not isinstance(nan_replace_value, (int, float)):
        raise ValueError(
            "Argument `nan_replace` is expected to be of a type `int` or `float` when `nan_strategy = 'replace`, "
            f"but got {nan_replace_value}"
        )


def conditional_entropy(
    preds: Tensor,
    target: Tensor,
    log_base: float = math.e,
    nan_strategy: Literal["replace", "drop"] = "replace",
    nan_replace_value: Optional[Union[int, float]] = 0.0,
) -> Tensor:
    r"""Computes conditional entropy of x given y: S(x|y)
    The implementation is based on the code provided in https://github.com/shakedzy/dython/

    Args:
        preds: 1D or 2D tensor of categorical (nominal) data
            - 1D shape: (batch_size,)
            - 2D shape: (batch_size, num_classes)
        target: 1D or 2D tensor of categorical (nominal) data
            - 1D shape: (batch_size,)
            - 2D shape: (batch_size, num_classes)
        log_base: float value to be used as the base for logarithm
        nan_strategy: Indication of whether to replace or drop ``NaN`` values
        nan_replace_value: Value to replace ``NaN``s when ``nan_strategy = 'replace'``

    Returns:
        S(x|y): float

    Example:
        >>> from torchmetrics.functional import conditional_entropy
        >>> _ = torch.manual_seed(42)
        >>> preds = torch.randint(10, (10,))
        >>> target = torch.randint(10, (10,))
        >>> conditional_entropy(preds, target)
        tensor(0.2773)
    """
    # reduce dim if ndim == 2
    preds = preds.argmax(1) if preds.ndim == 2 else preds
    target = target.argmax(1) if target.ndim == 2 else target
    # handle the nan in data
    preds, target = _handle_nan_in_data(preds, target, nan_strategy, nan_replace_value)

    # set x and y
    x, y = preds.numpy(), target.numpy()

    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x, y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0.0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y / p_xy, log_base)
    return torch.tensor(entropy)


def theils_u(
    preds: Tensor,
    target: Tensor,
    nan_strategy: Literal["replace", "drop"] = "replace",
    nan_replace_value: Optional[Union[int, float]] = 0.0,
    _PRECISION: float = 1e-13,
) -> Tensor:
    r"""Computes Theil's U Statistic (Uncertainty Coefficient). The value is
    between 0 and 1, i.e. 0 means y has no information about x while value 1
    means y has complete information about x.

    The implementation is based on the code provided in https://github.com/shakedzy/dython/

    Args:
        preds: 1D or 2D tensor of categorical (nominal) data
            - 1D shape: (batch_size,)
            - 2D shape: (batch_size, num_classes)
        target: 1D or 2D tensor of categorical (nominal) data
            - 1D shape: (batch_size,)
            - 2D shape: (batch_size, num_classes)
        nan_strategy: Indication of whether to replace or drop ``NaN`` values
        nan_replace_value: Value to replace ``NaN``s when ``nan_strategy = 'replace'``
        _PRECISION: used to round off values above 1 + _PRECISION and below 0 - _PRECISION
    Returns:
        Thiel's U Statistic: Tensor

    Example:
        >>> from torchmetrics.functional import theils_u
        >>> _ = torch.manual_seed(42)
        >>> preds = torch.randint(10, (10,))
        >>> target = torch.randint(10, (10,))
        >>> theils_u(preds, target)
        tensor(0.8530)
    """
    # reduce dim if ndim == 2
    preds = preds.argmax(1) if preds.ndim == 2 else preds
    target = target.argmax(1) if target.ndim == 2 else target
    # handle the nan in data
    preds, target = _handle_nan_in_data(preds, target, nan_strategy, nan_replace_value)

    # compute conditional entropy
    s_xy = conditional_entropy(preds, target)

    x_counter = Counter(preds.numpy())
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n / total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)

    # if s_x is 0 (all elements are same, entropy 0), return 1.0
    if s_x == 0:
        return 1.0

    # compute u statistic
    u = (s_x - s_xy) / s_x

    # compute precision issue
    if -_PRECISION <= u < 0.0 or 1.0 < u <= 1.0 + _PRECISION:
        rounded_u = 0.0 if u < 0 else 1.0
        warnings.warn(
            f"Rounded U = {u} to {rounded_u}. This is \
            probably due to floating point precision issues.",
            RuntimeWarning,
        )
        return rounded_u

    return u
