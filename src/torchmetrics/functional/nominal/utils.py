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
from typing import Optional

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.utilities.prints import rank_zero_warn


def _nominal_input_validation(nan_strategy: str, nan_replace_value: Optional[float]) -> None:
    if nan_strategy not in ["replace", "drop"]:
        raise ValueError(
            f"Argument `nan_strategy` is expected to be one of `['replace', 'drop']`, but got {nan_strategy}"
        )
    if nan_strategy == "replace" and not isinstance(nan_replace_value, (float, int)):
        raise ValueError(
            "Argument `nan_replace` is expected to be of a type `int` or `float` when `nan_strategy = 'replace`, "
            f"but got {nan_replace_value}"
        )


def _compute_expected_freqs(confmat: Tensor) -> Tensor:
    """Compute the expected frequenceis from the provided confusion matrix."""
    margin_sum_rows, margin_sum_cols = confmat.sum(1), confmat.sum(0)
    return torch.einsum("r, c -> rc", margin_sum_rows, margin_sum_cols) / confmat.sum()


def _compute_chi_squared(confmat: Tensor, bias_correction: bool) -> Tensor:
    """Chi-square test of independenc of variables in a confusion matrix table.

    Adapted from: https://github.com/scipy/scipy/blob/v1.9.2/scipy/stats/contingency.py.

    """
    expected_freqs = _compute_expected_freqs(confmat)
    # Get degrees of freedom
    df = expected_freqs.numel() - sum(expected_freqs.shape) + expected_freqs.ndim - 1
    if df == 0:
        return torch.tensor(0.0, device=confmat.device)

    if df == 1 and bias_correction:
        diff = expected_freqs - confmat
        direction = diff.sign()
        confmat += direction * torch.minimum(0.5 * torch.ones_like(direction), direction.abs())

    return torch.sum((confmat - expected_freqs) ** 2 / expected_freqs)


def _drop_empty_rows_and_cols(confmat: Tensor) -> Tensor:
    """Drop all rows and columns containing only zeros.

    Example:
        >>> from torch import randint
        >>> from torchmetrics.functional.nominal.utils import _drop_empty_rows_and_cols
        >>> matrix = randint(10, size=(4, 3))
        >>> matrix[1, :] = matrix[:, 1] = 0
        >>> matrix
        tensor([[2, 0, 6],
                [0, 0, 0],
                [0, 0, 0],
                [3, 0, 4]])
        >>> _drop_empty_rows_and_cols(matrix)
        tensor([[2, 6],
                [3, 4]])

    """
    confmat = confmat[confmat.sum(1) != 0]
    return confmat[:, confmat.sum(0) != 0]


def _compute_phi_squared_corrected(
    phi_squared: Tensor,
    num_rows: int,
    num_cols: int,
    confmat_sum: Tensor,
) -> Tensor:
    """Compute bias-corrected Phi Squared."""
    return torch.max(
        torch.tensor(0.0, device=phi_squared.device),
        phi_squared - ((num_rows - 1) * (num_cols - 1)) / (confmat_sum - 1),
    )


def _compute_rows_and_cols_corrected(num_rows: int, num_cols: int, confmat_sum: Tensor) -> tuple[Tensor, Tensor]:
    """Compute bias-corrected number of rows and columns."""
    rows_corrected = num_rows - (num_rows - 1) ** 2 / (confmat_sum - 1)
    cols_corrected = num_cols - (num_cols - 1) ** 2 / (confmat_sum - 1)
    return rows_corrected, cols_corrected


def _compute_bias_corrected_values(
    phi_squared: Tensor, num_rows: int, num_cols: int, confmat_sum: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute bias-corrected Phi Squared and number of rows and columns."""
    phi_squared_corrected = _compute_phi_squared_corrected(phi_squared, num_rows, num_cols, confmat_sum)
    rows_corrected, cols_corrected = _compute_rows_and_cols_corrected(num_rows, num_cols, confmat_sum)
    return phi_squared_corrected, rows_corrected, cols_corrected


def _handle_nan_in_data(
    preds: Tensor,
    target: Tensor,
    nan_strategy: Literal["replace", "drop"] = "replace",
    nan_replace_value: Optional[float] = 0.0,
) -> tuple[Tensor, Tensor]:
    """Handle ``NaN`` values in input data.

    If ``nan_strategy = 'replace'``, all ``NaN`` values are replaced with ``nan_replace_value``.
    If ``nan_strategy = 'drop'``, all rows containing ``NaN`` in any of two vectors are dropped.

    Args:
        preds: 1D tensor of categorical (nominal) data
        target: 1D tensor of categorical (nominal) data
        nan_strategy: Indication of whether to replace or drop ``NaN`` values
        nan_replace_value: Value to replace ``NaN`s when ``nan_strategy = 'replace```

    Returns:
        Updated ``preds`` and ``target`` tensors which contain no ``Nan``

    Raises:
        ValueError: If ``nan_strategy`` is not from ``['replace', 'drop']``.
        ValueError: If ``nan_strategy = replace`` and ``nan_replace_value`` is not of a type ``int`` or ``float``.

    """
    if nan_strategy == "replace":
        return preds.nan_to_num(nan_replace_value), target.nan_to_num(nan_replace_value)
    rows_contain_nan = torch.logical_or(preds.isnan(), target.isnan())
    return preds[~rows_contain_nan], target[~rows_contain_nan]


def _unable_to_use_bias_correction_warning(metric_name: str) -> None:
    rank_zero_warn(
        f"Unable to compute {metric_name} using bias correction. Please consider to set `bias_correction=False`."
    )
