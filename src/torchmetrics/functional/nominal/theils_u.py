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
import itertools
from typing import Optional, Union

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.classification.confusion_matrix import _multiclass_confusion_matrix_update
from torchmetrics.functional.nominal.utils import (
    _drop_empty_rows_and_cols,
    _handle_nan_in_data,
    _nominal_input_validation,
)


def _conditional_entropy_compute(confmat: Tensor) -> Tensor:
    r"""Compute Conditional Entropy Statistic based on a pre-computed confusion matrix.

    .. math::
        H(X|Y) = \sum_{x, y ~ (X, Y)} p(x, y)\frac{p(y)}{p(x, y)}

    Args:
        confmat: Confusion matrix for observed data

    Returns:
        Conditional Entropy Value
    """
    confmat = _drop_empty_rows_and_cols(confmat)
    total_occurrences = confmat.sum()
    # iterate over all i, j combinations
    p_xy_m = confmat / total_occurrences
    # get p_y by summing over x dim (=1)
    p_y = confmat.sum(1) / total_occurrences
    # repeat over rows (shape = p_xy_m.shape[1]) for tensor multiplication
    p_y_m = p_y.unsqueeze(1).repeat(1, p_xy_m.shape[1])

    # entropy calculated as p_xy * log (p_xy / p_y)
    return torch.nansum(p_xy_m * torch.log(p_y_m / p_xy_m))


def _theils_u_update(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    nan_strategy: Literal["replace", "drop"] = "replace",
    nan_replace_value: Optional[Union[int, float]] = 0.0,
) -> Tensor:
    """Compute the bins to update the confusion matrix with for Theil's U calculation.

    Args:
        preds: 1D or 2D tensor of categorical (nominal) data
        target: 1D or 2D tensor of categorical (nominal) data
        num_classes: Integer specifing the number of classes
        nan_strategy: Indication of whether to replace or drop ``NaN`` values
        nan_replace_value: Value to replace ``NaN`s when ``nan_strategy = 'replace```

    Returns:
        Non-reduced confusion matrix
    """
    preds = preds.argmax(1) if preds.ndim == 2 else preds
    target = target.argmax(1) if target.ndim == 2 else target
    preds, target = _handle_nan_in_data(preds, target, nan_strategy, nan_replace_value)
    return _multiclass_confusion_matrix_update(preds, target, num_classes)


def _theils_u_compute(confmat: Tensor) -> Tensor:
    """Compute Theil's U statistic based on a pre-computed confusion matrix.

    Args:
        confmat: Confusion matrix for observed data

    Returns:
        Theil's U statistic
    """
    confmat = _drop_empty_rows_and_cols(confmat)

    # compute conditional entropy
    s_xy = _conditional_entropy_compute(confmat)

    # compute H(x)
    total_occurrences = confmat.sum()
    p_x = confmat.sum(0) / total_occurrences
    s_x = -torch.sum(p_x * torch.log(p_x))

    # compute u statistic
    if s_x == 0:
        return torch.tensor(0, device=confmat.device)

    return (s_x - s_xy) / s_x


def theils_u(
    preds: Tensor,
    target: Tensor,
    nan_strategy: Literal["replace", "drop"] = "replace",
    nan_replace_value: Optional[Union[int, float]] = 0.0,
) -> Tensor:
    r"""Compute `Theil's U`_ statistic (Uncertainty Coef.) measuring the association between two nominal data series.

    .. math::
        U(X|Y) = \frac{H(X) - H(X|Y)}{H(X)}

    where :math:`H(X)` is entropy of variable :math:`X` while :math:`H(X|Y)` is the conditional entropy of :math:`X`
    given :math:`Y`.

    Theils's U is an asymmetric coefficient, i.e. :math:`TheilsU(preds, target) \neq TheilsU(target, preds)`.

    The output values lies in [0, 1]. 0 means y has no information about x while value 1 means y has complete
    information about x.

    Args:
        preds: 1D or 2D tensor of categorical (nominal) data
            - 1D shape: (batch_size,)
            - 2D shape: (batch_size, num_classes)
        target: 1D or 2D tensor of categorical (nominal) data
            - 1D shape: (batch_size,)
            - 2D shape: (batch_size, num_classes)
        nan_strategy: Indication of whether to replace or drop ``NaN`` values
        nan_replace_value: Value to replace ``NaN``s when ``nan_strategy = 'replace'``

    Returns:
        Theil's U Statistic: Tensor

    Example:
        >>> from torchmetrics.functional.nominal import theils_u
        >>> _ = torch.manual_seed(42)
        >>> preds = torch.randint(10, (10,))
        >>> target = torch.randint(10, (10,))
        >>> theils_u(preds, target)
        tensor(0.8530)
    """
    num_classes = len(torch.cat([preds, target]).unique())
    confmat = _theils_u_update(preds, target, num_classes, nan_strategy, nan_replace_value)
    return _theils_u_compute(confmat)


def theils_u_matrix(
    matrix: Tensor,
    nan_strategy: Literal["replace", "drop"] = "replace",
    nan_replace_value: Optional[Union[int, float]] = 0.0,
) -> Tensor:
    r"""Compute `Theil's U`_ statistic between a set of multiple variables.

    This can serve as a convenient tool to compute Theil's U statistic for analyses of correlation between categorical
    variables in your dataset.

    Args:
        matrix: A tensor of categorical (nominal) data, where:
            - rows represent a number of data points
            - columns represent a number of categorical (nominal) features
        nan_strategy: Indication of whether to replace or drop ``NaN`` values
        nan_replace_value: Value to replace ``NaN``s when ``nan_strategy = 'replace'``

    Returns:
        Theil's U statistic for a dataset of categorical variables

    Example:
        >>> from torchmetrics.functional.nominal import theils_u_matrix
        >>> _ = torch.manual_seed(42)
        >>> matrix = torch.randint(0, 4, (200, 5))
        >>> theils_u_matrix(matrix)
        tensor([[1.0000, 0.0202, 0.0142, 0.0196, 0.0353],
                [0.0202, 1.0000, 0.0070, 0.0136, 0.0065],
                [0.0143, 0.0070, 1.0000, 0.0125, 0.0206],
                [0.0198, 0.0137, 0.0125, 1.0000, 0.0312],
                [0.0352, 0.0065, 0.0204, 0.0308, 1.0000]])
    """
    _nominal_input_validation(nan_strategy, nan_replace_value)
    num_variables = matrix.shape[1]
    theils_u_matrix_value = torch.ones(num_variables, num_variables, device=matrix.device)
    for i, j in itertools.combinations(range(num_variables), 2):
        x, y = matrix[:, i], matrix[:, j]
        num_classes = len(torch.cat([x, y]).unique())
        confmat = _theils_u_update(x, y, num_classes, nan_strategy, nan_replace_value)
        theils_u_matrix_value[i, j] = _theils_u_compute(confmat)
        theils_u_matrix_value[j, i] = _theils_u_compute(confmat.T)
    return theils_u_matrix_value
