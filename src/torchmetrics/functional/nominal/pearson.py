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
    _compute_chi_squared,
    _drop_empty_rows_and_cols,
    _handle_nan_in_data,
    _nominal_input_validation,
)


def _pearsons_contingency_coefficient_update(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    nan_strategy: Literal["replace", "drop"] = "replace",
    nan_replace_value: Optional[Union[int, float]] = 0.0,
) -> Tensor:
    """Compute the bins to update the confusion matrix with for Pearson's Contingency Coefficient calculation.

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


def _pearsons_contingency_coefficient_compute(confmat: Tensor) -> Tensor:
    """Compute Pearson's Contingency Coefficient based on a pre-computed confusion matrix.

    Args:
        confmat: Confusion matrix for observed data

    Returns:
        Pearson's Contingency Coefficient
    """
    confmat = _drop_empty_rows_and_cols(confmat)
    cm_sum = confmat.sum()
    chi_squared = _compute_chi_squared(confmat, bias_correction=False)
    phi_squared = chi_squared / cm_sum

    tschuprows_t_value = torch.sqrt(phi_squared / (1 + phi_squared))
    return tschuprows_t_value.clamp(0.0, 1.0)


def pearsons_contingency_coefficient(
    preds: Tensor,
    target: Tensor,
    nan_strategy: Literal["replace", "drop"] = "replace",
    nan_replace_value: Optional[Union[int, float]] = 0.0,
) -> Tensor:
    r"""Compute `Pearson's Contingency Coefficient`_ for measuring the association between two categorical data series.

    .. math::
        Pearson = \sqrt{\frac{\chi^2 / n}{1 + \chi^2 / n}}

    where

    .. math::
        \chi^2 = \sum_{i,j} \ frac{\left(n_{ij} - \frac{n_{i.} n_{.j}}{n}\right)^2}{\frac{n_{i.} n_{.j}}{n}}

    where :math:`n_{ij}` denotes the number of times the values :math:`(A_i, B_j)` are observed with :math:`A_i, B_j`
    represent frequencies of values in ``preds`` and ``target``, respectively.

    Pearson's Contingency Coefficient is a symmetric coefficient, i.e.
    :math:`Pearson(preds, target) = Pearson(target, preds)`.

    The output values lies in [0, 1] with 1 meaning the perfect association.

    Args:
        preds: 1D or 2D tensor of categorical (nominal) data:

            - 1D shape: (batch_size,)
            - 2D shape: (batch_size, num_classes)

        target: 1D or 2D tensor of categorical (nominal) data:

            - 1D shape: (batch_size,)
            - 2D shape: (batch_size, num_classes)

        nan_strategy: Indication of whether to replace or drop ``NaN`` values
        nan_replace_value: Value to replace ``NaN``s when ``nan_strategy = 'replace'``

    Returns:
        Pearson's Contingency Coefficient

    Example:
        >>> from torchmetrics.functional.nominal import pearsons_contingency_coefficient
        >>> _ = torch.manual_seed(42)
        >>> preds = torch.randint(0, 4, (100,))
        >>> target = torch.round(preds + torch.randn(100)).clamp(0, 4)
        >>> pearsons_contingency_coefficient(preds, target)
        tensor(0.6948)
    """
    _nominal_input_validation(nan_strategy, nan_replace_value)
    num_classes = len(torch.cat([preds, target]).unique())
    confmat = _pearsons_contingency_coefficient_update(preds, target, num_classes, nan_strategy, nan_replace_value)
    return _pearsons_contingency_coefficient_compute(confmat)


def pearsons_contingency_coefficient_matrix(
    matrix: Tensor,
    nan_strategy: Literal["replace", "drop"] = "replace",
    nan_replace_value: Optional[Union[int, float]] = 0.0,
) -> Tensor:
    r"""Compute `Pearson's Contingency Coefficient`_ statistic between a set of multiple variables.

    This can serve as a convenient tool to compute Pearson's Contingency Coefficient for analyses
    of correlation between categorical variables in your dataset.

    Args:
        matrix: A tensor of categorical (nominal) data, where:

            - rows represent a number of data points
            - columns represent a number of categorical (nominal) features

        nan_strategy: Indication of whether to replace or drop ``NaN`` values
        nan_replace_value: Value to replace ``NaN``s when ``nan_strategy = 'replace'``

    Returns:
        Pearson's Contingency Coefficient statistic for a dataset of categorical variables

    Example:
        >>> from torchmetrics.functional.nominal import pearsons_contingency_coefficient_matrix
        >>> _ = torch.manual_seed(42)
        >>> matrix = torch.randint(0, 4, (200, 5))
        >>> pearsons_contingency_coefficient_matrix(matrix)
        tensor([[1.0000, 0.2326, 0.1959, 0.2262, 0.2989],
                [0.2326, 1.0000, 0.1386, 0.1895, 0.1329],
                [0.1959, 0.1386, 1.0000, 0.1840, 0.2335],
                [0.2262, 0.1895, 0.1840, 1.0000, 0.2737],
                [0.2989, 0.1329, 0.2335, 0.2737, 1.0000]])
    """
    _nominal_input_validation(nan_strategy, nan_replace_value)
    num_variables = matrix.shape[1]
    pearsons_cont_coef_matrix_value = torch.ones(num_variables, num_variables, device=matrix.device)
    for i, j in itertools.combinations(range(num_variables), 2):
        x, y = matrix[:, i], matrix[:, j]
        num_classes = len(torch.cat([x, y]).unique())
        confmat = _pearsons_contingency_coefficient_update(x, y, num_classes, nan_strategy, nan_replace_value)
        val = _pearsons_contingency_coefficient_compute(confmat)
        pearsons_cont_coef_matrix_value[i, j] = pearsons_cont_coef_matrix_value[j, i] = val
    return pearsons_cont_coef_matrix_value
