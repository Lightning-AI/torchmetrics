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
from typing import Optional

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.classification.confusion_matrix import _multiclass_confusion_matrix_update
from torchmetrics.functional.nominal.utils import (
    _compute_bias_corrected_values,
    _compute_chi_squared,
    _drop_empty_rows_and_cols,
    _handle_nan_in_data,
    _nominal_input_validation,
    _unable_to_use_bias_correction_warning,
)


def _cramers_v_update(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    nan_strategy: Literal["replace", "drop"] = "replace",
    nan_replace_value: Optional[float] = 0.0,
) -> Tensor:
    """Compute the bins to update the confusion matrix with for Cramer's V calculation.

    Args:
        preds: 1D or 2D tensor of categorical (nominal) data
        target: 1D or 2D tensor of categorical (nominal) data
        num_classes: Integer specifying the number of classes
        nan_strategy: Indication of whether to replace or drop ``NaN`` values
        nan_replace_value: Value to replace ``NaN`s when ``nan_strategy = 'replace```

    Returns:
        Non-reduced confusion matrix

    """
    preds = preds.argmax(1) if preds.ndim == 2 else preds
    target = target.argmax(1) if target.ndim == 2 else target
    preds, target = _handle_nan_in_data(preds, target, nan_strategy, nan_replace_value)
    return _multiclass_confusion_matrix_update(preds, target, num_classes)


def _cramers_v_compute(confmat: Tensor, bias_correction: bool) -> Tensor:
    """Compute Cramers' V statistic based on a pre-computed confusion matrix.

    Args:
        confmat: Confusion matrix for observed data
        bias_correction: Indication of whether to use bias correction.

    Returns:
        Cramer's V statistic

    """
    confmat = _drop_empty_rows_and_cols(confmat)
    cm_sum = confmat.sum()
    chi_squared = _compute_chi_squared(confmat, bias_correction)
    phi_squared = chi_squared / cm_sum
    num_rows, num_cols = confmat.shape

    if bias_correction:
        phi_squared_corrected, rows_corrected, cols_corrected = _compute_bias_corrected_values(
            phi_squared, num_rows, num_cols, cm_sum
        )
        if torch.min(rows_corrected, cols_corrected) == 1:
            _unable_to_use_bias_correction_warning(metric_name="Cramer's V")
            return torch.tensor(float("nan"), device=confmat.device)
        cramers_v_value = torch.sqrt(phi_squared_corrected / torch.min(rows_corrected - 1, cols_corrected - 1))
    else:
        cramers_v_value = torch.sqrt(phi_squared / min(num_rows - 1, num_cols - 1))
    return cramers_v_value.clamp(0.0, 1.0)


def cramers_v(
    preds: Tensor,
    target: Tensor,
    bias_correction: bool = True,
    nan_strategy: Literal["replace", "drop"] = "replace",
    nan_replace_value: Optional[float] = 0.0,
) -> Tensor:
    r"""Compute `Cramer's V`_ statistic measuring the association between two categorical (nominal) data series.

    .. math::
        V = \sqrt{\frac{\chi^2 / n}{\min(r - 1, k - 1)}}

    where

    .. math::
        \chi^2 = \sum_{i,j} \ frac{\left(n_{ij} - \frac{n_{i.} n_{.j}}{n}\right)^2}{\frac{n_{i.} n_{.j}}{n}}

    where :math:`n_{ij}` denotes the number of times the values :math:`(A_i, B_j)` are observed with :math:`A_i, B_j`
    represent frequencies of values in ``preds`` and ``target``, respectively.

    Cramer's V is a symmetric coefficient, i.e. :math:`V(preds, target) = V(target, preds)`.

    The output values lies in [0, 1] with 1 meaning the perfect association.

    Args:
        preds: 1D or 2D tensor of categorical (nominal) data
            - 1D shape: (batch_size,)
            - 2D shape: (batch_size, num_classes)
        target: 1D or 2D tensor of categorical (nominal) data
            - 1D shape: (batch_size,)
            - 2D shape: (batch_size, num_classes)
        bias_correction: Indication of whether to use bias correction.
        nan_strategy: Indication of whether to replace or drop ``NaN`` values
        nan_replace_value: Value to replace ``NaN``s when ``nan_strategy = 'replace'``

    Returns:
        Cramer's V statistic

    Example:
        >>> from torch import randint, round
        >>> from torchmetrics.functional.nominal import cramers_v
        >>> preds = randint(0, 4, (100,))
        >>> target = round(preds + torch.randn(100)).clamp(0, 4)
        >>> cramers_v(preds, target)
        tensor(0.5284)

    """
    _nominal_input_validation(nan_strategy, nan_replace_value)
    num_classes = len(torch.cat([preds, target]).unique())
    confmat = _cramers_v_update(preds, target, num_classes, nan_strategy, nan_replace_value)
    return _cramers_v_compute(confmat, bias_correction)


def cramers_v_matrix(
    matrix: Tensor,
    bias_correction: bool = True,
    nan_strategy: Literal["replace", "drop"] = "replace",
    nan_replace_value: Optional[float] = 0.0,
) -> Tensor:
    r"""Compute `Cramer's V`_ statistic between a set of multiple variables.

    This can serve as a convenient tool to compute Cramer's V statistic for analyses of correlation between categorical
    variables in your dataset.

    Args:
        matrix: A tensor of categorical (nominal) data, where:
            - rows represent a number of data points
            - columns represent a number of categorical (nominal) features
        bias_correction: Indication of whether to use bias correction.
        nan_strategy: Indication of whether to replace or drop ``NaN`` values
        nan_replace_value: Value to replace ``NaN``s when ``nan_strategy = 'replace'``

    Returns:
        Cramer's V statistic for a dataset of categorical variables

    Example:
        >>> from torch import randint
        >>> from torchmetrics.functional.nominal import cramers_v_matrix
        >>> matrix = randint(0, 4, (200, 5))
        >>> cramers_v_matrix(matrix)
        tensor([[1.0000, 0.0637, 0.0000, 0.0542, 0.1337],
                [0.0637, 1.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 1.0000, 0.0000, 0.0649],
                [0.0542, 0.0000, 0.0000, 1.0000, 0.1100],
                [0.1337, 0.0000, 0.0649, 0.1100, 1.0000]])

    """
    _nominal_input_validation(nan_strategy, nan_replace_value)
    num_variables = matrix.shape[1]
    cramers_v_matrix_value = torch.ones(num_variables, num_variables, device=matrix.device)
    for i, j in itertools.combinations(range(num_variables), 2):
        x, y = matrix[:, i], matrix[:, j]
        num_classes = len(torch.cat([x, y]).unique())
        confmat = _cramers_v_update(x, y, num_classes, nan_strategy, nan_replace_value)
        cramers_v_matrix_value[i, j] = cramers_v_matrix_value[j, i] = _cramers_v_compute(confmat, bias_correction)
    return cramers_v_matrix_value
