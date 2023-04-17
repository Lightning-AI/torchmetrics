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
    _compute_bias_corrected_values,
    _compute_chi_squared,
    _drop_empty_rows_and_cols,
    _handle_nan_in_data,
    _nominal_input_validation,
    _unable_to_use_bias_correction_warning,
)


def _tschuprows_t_update(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    nan_strategy: Literal["replace", "drop"] = "replace",
    nan_replace_value: Optional[Union[int, float]] = 0.0,
) -> Tensor:
    """Compute the bins to update the confusion matrix with for Tschuprow's T calculation.

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


def _tschuprows_t_compute(confmat: Tensor, bias_correction: bool) -> Tensor:
    """Compute Tschuprow's T statistic based on a pre-computed confusion matrix.

    Args:
        confmat: Confusion matrix for observed data
        bias_correction: Indication of whether to use bias correction.

    Returns:
        Tschuprow's T statistic
    """
    confmat = _drop_empty_rows_and_cols(confmat)
    cm_sum = confmat.sum()
    chi_squared = _compute_chi_squared(confmat, bias_correction)
    phi_squared = chi_squared / cm_sum
    n_rows, n_cols = confmat.shape

    if bias_correction:
        phi_squared_corrected, rows_corrected, cols_corrected = _compute_bias_corrected_values(
            phi_squared, n_rows, n_cols, cm_sum
        )
        if torch.min(rows_corrected, cols_corrected) == 1:
            _unable_to_use_bias_correction_warning(metric_name="Tschuprow's T")
            return torch.tensor(float("nan"), device=confmat.device)
        tschuprows_t_value = torch.sqrt(phi_squared_corrected / torch.sqrt((rows_corrected - 1) * (cols_corrected - 1)))
    else:
        n_rows_tensor = torch.tensor(n_rows, device=phi_squared.device)
        n_cols_tensor = torch.tensor(n_cols, device=phi_squared.device)
        tschuprows_t_value = torch.sqrt(phi_squared / torch.sqrt((n_rows_tensor - 1) * (n_cols_tensor - 1)))
    return tschuprows_t_value.clamp(0.0, 1.0)


def tschuprows_t(
    preds: Tensor,
    target: Tensor,
    bias_correction: bool = True,
    nan_strategy: Literal["replace", "drop"] = "replace",
    nan_replace_value: Optional[Union[int, float]] = 0.0,
) -> Tensor:
    r"""Compute `Tschuprow's T`_ statistic measuring the association between two categorical (nominal) data series.

    .. math::
        T = \sqrt{\frac{\chi^2 / n}{\sqrt{(r - 1) * (k - 1)}}}

    where

    .. math::
        \chi^2 = \sum_{i,j} \ frac{\left(n_{ij} - \frac{n_{i.} n_{.j}}{n}\right)^2}{\frac{n_{i.} n_{.j}}{n}}

    where :math:`n_{ij}` denotes the number of times the values :math:`(A_i, B_j)` are observed with :math:`A_i, B_j`
    represent frequencies of values in ``preds`` and ``target``, respectively.

    Tschuprow's T is a symmetric coefficient, i.e. :math:`T(preds, target) = T(target, preds)`.

    The output values lies in [0, 1] with 1 meaning the perfect association.

    Args:
        preds: 1D or 2D tensor of categorical (nominal) data:

            - 1D shape: (batch_size,)
            - 2D shape: (batch_size, num_classes)

        target: 1D or 2D tensor of categorical (nominal) data:

            - 1D shape: (batch_size,)
            - 2D shape: (batch_size, num_classes)

        bias_correction: Indication of whether to use bias correction.
        nan_strategy: Indication of whether to replace or drop ``NaN`` values
        nan_replace_value: Value to replace ``NaN``s when ``nan_strategy = 'replace'``

    Returns:
        Tschuprow's T statistic

    Example:
        >>> from torchmetrics.functional.nominal import tschuprows_t
        >>> _ = torch.manual_seed(42)
        >>> preds = torch.randint(0, 4, (100,))
        >>> target = torch.round(preds + torch.randn(100)).clamp(0, 4)
        >>> tschuprows_t(preds, target)
        tensor(0.4930)
    """
    _nominal_input_validation(nan_strategy, nan_replace_value)
    num_classes = len(torch.cat([preds, target]).unique())
    confmat = _tschuprows_t_update(preds, target, num_classes, nan_strategy, nan_replace_value)
    return _tschuprows_t_compute(confmat, bias_correction)


def tschuprows_t_matrix(
    matrix: Tensor,
    bias_correction: bool = True,
    nan_strategy: Literal["replace", "drop"] = "replace",
    nan_replace_value: Optional[Union[int, float]] = 0.0,
) -> Tensor:
    r"""Compute `Tschuprow's T`_ statistic between a set of multiple variables.

    This can serve as a convenient tool to compute Tschuprow's T statistic for analyses of correlation between
    categorical variables in your dataset.

    Args:
        matrix: A tensor of categorical (nominal) data, where:

            - rows represent a number of data points
            - columns represent a number of categorical (nominal) features

        bias_correction: Indication of whether to use bias correction.
        nan_strategy: Indication of whether to replace or drop ``NaN`` values
        nan_replace_value: Value to replace ``NaN``s when ``nan_strategy = 'replace'``

    Returns:
        Tschuprow's T statistic for a dataset of categorical variables

    Example:
        >>> from torchmetrics.functional.nominal import tschuprows_t_matrix
        >>> _ = torch.manual_seed(42)
        >>> matrix = torch.randint(0, 4, (200, 5))
        >>> tschuprows_t_matrix(matrix)
        tensor([[1.0000, 0.0637, 0.0000, 0.0542, 0.1337],
                [0.0637, 1.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 1.0000, 0.0000, 0.0649],
                [0.0542, 0.0000, 0.0000, 1.0000, 0.1100],
                [0.1337, 0.0000, 0.0649, 0.1100, 1.0000]])
    """
    _nominal_input_validation(nan_strategy, nan_replace_value)
    num_variables = matrix.shape[1]
    tschuprows_t_matrix_value = torch.ones(num_variables, num_variables, device=matrix.device)
    for i, j in itertools.combinations(range(num_variables), 2):
        x, y = matrix[:, i], matrix[:, j]
        num_classes = len(torch.cat([x, y]).unique())
        confmat = _tschuprows_t_update(x, y, num_classes, nan_strategy, nan_replace_value)
        tschuprows_t_matrix_value[i, j] = tschuprows_t_matrix_value[j, i] = _tschuprows_t_compute(
            confmat, bias_correction
        )
    return tschuprows_t_matrix_value
