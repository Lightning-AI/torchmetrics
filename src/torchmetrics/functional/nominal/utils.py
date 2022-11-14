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
from typing_extensions import Literal


def _handle_nan_in_data(
    preds: Tensor,
    target: Tensor,
    nan_strategy: Literal["replace", "drop"] = "replace",
    nan_replace_value: Optional[float] = 0.0,
) -> Tuple[Tensor, Tensor]:
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
