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

    Raises:
        ValueError: If ``nan_strategy`` is not from ``['replace', 'drop']``.
        ValueError: If ``nan_strategy = replace`` and ``nan_replace_value`` is not of a type ``int`` or ``float``.
    """
    if nan_strategy == "replace":
        return preds.nan_to_num(nan_replace_value), target.nan_to_num(nan_replace_value)
    rows_contain_nan = torch.logical_or(preds.isnan(), target.isnan())
    return preds[~rows_contain_nan], target[~rows_contain_nan]
