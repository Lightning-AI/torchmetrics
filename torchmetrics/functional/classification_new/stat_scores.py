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

from torchmetrics.utilities.checks import _check_same_shape


def _binary_stat_scores_arg_validation(
    threshold: float = 0.5,
    multidim_average: str = "global",
    ignore_index: Optional[int] = None,
) -> None:
    """Validate non tensor input."""
    if not isinstance(threshold, float):
        raise ValueError(f"Expected argument `threshold` to be a float, but got {threshold}.")
    allowed_multidim_average = ("global", "samplewise")
    if not isinstance(multidim_average, str) and multidim_average not in allowed_multidim_average:
        raise ValueError(
            f"Expected argument `multidim_average` to be one of {allowed_multidim_average}, but got {multidim_average}"
        )
    if ignore_index is not None and not isinstance(ignore_index, int):
        raise ValueError(f"Expected argument `ignore_index` to either be `None` or an integer, but got {ignore_index}")


def _binary_stat_scores_tensor_validation(
    preds: Tensor, target: Tensor, multidim_average: str = "global", ignore_index: Optional[int] = bool
) -> None:
    """Validate tensor input."""
    # Check that they have same shape
    _check_same_shape(preds, target)

    # Check that target only contains [0,1] values or value in ignore_index
    unique_values = torch.unique(target)
    if ignore_index is None:
        check = torch.any((unique_values != 0) & (unique_values != 1))
    else:
        check = torch.any((unique_values != 0) & (unique_values != 1) & (unique_values != ignore_index))
    if check:
        raise RuntimeError(
            "Detected the following values in `target`: {unique_values} but expected only"
            " the following values {[0,1] + [] if ignore_index is None else [ignore_index]}."
        )

    # If preds is label tensor, also check that it only contains [0,1] values
    if not preds.is_floating_point():
        unique_values = torch.unique(preds)
        if torch.any((unique_values != 0) & (unique_values != 1)):
            raise RuntimeError(
                "Detected the following values in `preds`: {unique_values} but expected only"
                " the following values [0,1] since preds is a label tensor."
            )

    if multidim_average != "global" and preds.ndim < 2:
        raise ValueError("Expected input to be atleast 2D when multidim_average is set to `samplewise`")


def _binary_stat_scores_format(
    preds: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """Convert all input to label format."""
    if preds.is_floating_point():
        if not ((0 <= preds) * (preds <= 1)).all():
            # preds is logits, convert with sigmoid
            preds = preds.sigmoid()
        preds = preds > threshold

    preds = preds.reshape(preds.shape[0], -1)
    target = target.reshape(target.shape[0], -1)

    if ignore_index is not None:
        idx = target == ignore_index
        target[idx] = -1

    return preds, target


def _binary_stat_scores_update(
    preds: Tensor,
    target: Tensor,
    multidim_average: str = "global",
) -> Tensor:
    """"""
    sum_dim = 0 if multidim_average == "global" else 1
    tp = ((target == preds) & (target == 1)).sum(sum_dim).squeeze()
    fn = ((target != preds) & (target == 1)).sum(sum_dim).squeeze()
    fp = ((target != preds) & (target == 0)).sum(sum_dim).squeeze()
    tn = ((target == preds) & (target == 0)).sum(sum_dim).squeeze()
    return tp, fp, tn, fn


def _binary_stat_scores_compute(
    tp: Tensor, fp: Tensor, tn: Tensor, fn: Tensor, multidim_average: str = "global"
) -> Tensor:
    if multidim_average == "global":
        return torch.cat([tp, fp, tn, fn, tp + fp + tn + fn], dim=0)
    return torch.stack([tp, fp, tn, fn, tp + fp + tn + fn], dim=1)


def binary_stat_scores(
    preds: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    multidim_average: str = "global",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    if validate_args:
        _binary_stat_scores_arg_validation(threshold, multidim_average, ignore_index)
        _binary_stat_scores_tensor_validation(preds, target, multidim_average, ignore_index)
    preds, target = _binary_stat_scores_format(preds, target, threshold, ignore_index)
    tp, fp, tn, fn = _binary_stat_scores_update(preds, target, multidim_average)
    return _binary_stat_scores_compute(tp, fp, tn, fn, multidim_average)
