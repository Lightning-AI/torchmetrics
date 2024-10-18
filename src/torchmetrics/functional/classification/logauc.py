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
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.classification.roc import binary_roc, multiclass_roc, multilabel_roc
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.compute import _auc_compute_without_check


def _interpolate(newpoints: Tensor, x: Tensor, y: Tensor) -> Tensor:
    """Interpolate the points (x, y) to the newpoints using linear interpolation."""
    # TODO: Add native torch implementation
    device = newpoints.device
    newpoints_n = newpoints.cpu().numpy()
    x_n = x.cpu().numpy()
    y_n = y.cpu().numpy()
    return torch.from_numpy(np.interp(newpoints_n, x_n, y_n)).to(device)


def _validate_fpr_range(fpr_range: Tuple[float, float]) -> None:
    if not isinstance(fpr_range, tuple) and not len(fpr_range) == 2:
        raise ValueError(f"The `fpr_range` should be a tuple of two floats, but got {type(fpr_range)}.")
    if not (0 <= fpr_range[0] < fpr_range[1] <= 1):
        raise ValueError(f"The `fpr_range` should be a tuple of two floats in the range [0, 1], but got {fpr_range}.")


def _binary_logauc_compute(
    fpr: Tensor,
    tpr: Tensor,
    fpr_range: Tuple[float, float] = (0.001, 0.1),
) -> Tensor:
    fpr_range = torch.tensor(fpr_range).to(fpr.device)
    if fpr.numel() < 2 or tpr.numel() < 2:
        rank_zero_warn(
            "At least two values on for the fpr and tpr are required to compute the log AUC. Returns 0 score."
        )
        return torch.tensor(0.0, device=fpr.device)

    tpr = torch.cat([tpr, _interpolate(fpr_range, fpr, tpr)]).sort().values
    fpr = torch.cat([fpr, fpr_range]).sort().values

    log_fpr = torch.log10(fpr)
    bounds = torch.log10(torch.tensor(fpr_range))

    lower_bound_idx = torch.where(log_fpr == bounds[0])[0][-1]
    upper_bound_idx = torch.where(log_fpr == bounds[1])[0][-1]

    trimmed_log_fpr = log_fpr[lower_bound_idx : upper_bound_idx + 1]
    trimmed_tpr = tpr[lower_bound_idx : upper_bound_idx + 1]

    # compute area and rescale it to the range of fpr
    return _auc_compute_without_check(trimmed_log_fpr, trimmed_tpr, 1.0) / (bounds[1] - bounds[0])


def binary_logauc(
    preds: Tensor,
    target: Tensor,
    fpr_range: Tuple[float, float] = (0.001, 0.1),
    thresholds: Optional[Union[int, List[float], Tensor]] = None,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    _validate_fpr_range(fpr_range)
    fpr, tpr, _ = binary_roc(preds, target, thresholds, ignore_index, validate_args)
    return _binary_logauc_compute(fpr, tpr, fpr_range)


def _multiclass_logauc_compute(
    fpr: Union[Tensor, List[Tensor]],
    tpr: Union[Tensor, List[Tensor]],
    fpr_range: Tuple[float, float] = (0.001, 0.1),
    average: Optional[Literal["macro", "none"]] = "macro",
) -> Tensor:
    scores = []
    for fpr_i, tpr_i in zip(fpr, tpr):
        scores.append(_binary_logauc_compute(fpr_i, tpr_i, fpr_range))
    scores = torch.stack(scores)
    if average == "macro":
        return scores.mean()
    return scores


def multiclass_logauc(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    fpr_range: Tuple[float, float] = (0.001, 0.1),
    average: Optional[Literal["macro", "weighted", "none"]] = "macro",
    thresholds: Optional[Union[int, List[float], Tensor]] = None,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    _validate_fpr_range(fpr_range)
    fpr, tpr, _ = multiclass_roc(
        preds, target, num_classes, thresholds, average=None, ignore_index=ignore_index, validate_args=validate_args
    )
    return _multiclass_logauc_compute(fpr, tpr, fpr_range, average)


def _multilabel_logauc_compute(
    fpr: Union[Tensor, List[Tensor]],
    tpr: Union[Tensor, List[Tensor]],
    fpr_range: Tuple[float, float] = (0.001, 0.1),
    average: Optional[Literal["macro", "weighted", "none"]] = "macro",
) -> Tensor:
    pass


def multilabel_logauc(
    preds: Tensor,
    target: Tensor,
    num_labels: int,
    thresholds: Optional[Union[int, List[float], Tensor]] = None,
    fpr_range: Tuple[float, float] = (0.001, 0.1),
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    fpr, tpr, _ = multilabel_roc(preds, target, num_labels, thresholds, ignore_index, validate_args)
    return _multilabel_logauc_compute(fpr, tpr, fpr_range)


def logauc() -> Tensor:
    pass
