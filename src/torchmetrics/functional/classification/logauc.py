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
from torchmetrics.utilities.compute import _auc_compute_without_check


def _interpolate(newpoints: Tensor, x: Tensor, y: Tensor) -> Tensor:
    """Interpolate the points (x, y) to the newpoints using linear interpolation."""
    # TODO: Add native torch implementation
    device = newpoints.device
    newpoints_n = newpoints.cpu().numpy()
    x_n = x.cpu().numpy()
    y_n = y.cpu().numpy()
    return torch.from_numpy(np.interp(newpoints_n, x_n, y_n)).to(device)


def _binary_logauc_compute(
    fpr: Tensor,
    tpr: Tensor,
    fpr_range: Tuple[float, float] = (0.001, 0.1),
) -> Tensor:
    fpr_range = torch.tensor(fpr_range).to(fpr.device)
    tpr = torch.cat([tpr, _interpolate(fpr_range, fpr, tpr)]).sort().values
    fpr = torch.cat([fpr, fpr_range]).sort().values

    log_fpr = torch.log10(fpr)
    bounds = torch.log10(torch.tensor(fpr_range))

    lower_bound_idx = torch.where(log_fpr == bounds[0])[0][-1]
    upper_bound_idx = torch.where(log_fpr == bounds[1])[0][-1]

    trimmed_log_fpr = log_fpr[lower_bound_idx : upper_bound_idx + 1]
    trimmed_tpr = tpr[lower_bound_idx : upper_bound_idx + 1]

    # compute area and rescale it to the range of fpr
    area = _auc_compute_without_check(trimmed_log_fpr, trimmed_tpr, 1.0) / (bounds[1] - bounds[0])
    return area


def binary_logauc(
    preds: Tensor,
    target: Tensor,
    thresholds: Optional[Union[int, List[float], Tensor]] = None,
    fpr_range: Tuple[float, float] = (0.001, 0.1),
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    fpr, tpr, _ = binary_roc(preds, target, thresholds, ignore_index, validate_args)
    return _binary_logauc_compute(fpr, tpr, fpr_range)


def _multiclass_logauc_compute() -> Tensor:
    pass


def multiclass_logauc(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    thresholds: Optional[Union[int, List[float], Tensor]] = None,
    average: Optional[Literal["micro", "macro"]] = None,
    fpr_range: Tuple[float, float] = (0.001, 0.1),
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    fpr, tpr, _ = multiclass_roc(preds, target, num_classes, thresholds, average, ignore_index, validate_args)
    return _multiclass_logauc_compute(fpr, tpr, fpr_range)


def _multilabel_logauc_compute() -> Tensor:
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
