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
from torchmetrics.functional.classification.roc import binary_roc, multiclass_roc, multilabel_roc


from torchmetrics.utilities.compute import _auc_compute_without_check
from typing import Union, Optional, Tuple, List
from torch import Tensor
import torch
import numpy as np
from typing_extensions import Literal


def _interpolate(newpoints: Tensor, x: Tensor, y: Tensor) -> Tensor:
    """Interpolate the points (x, y) to the newpoints using linear interpolation."""
    # TODO: Add native torch implementation
    return torch.from_numpy(np.interp(newpoints.numpy(), x.numpy(), y.numpy()))


def _binary_logauc_compute(
    fpr: Tensor,
    tpr: Tensor,
    fpr_range: Tuple[float, float] = (0.001, 0.1),
) -> Tensor:
    tpr = torch.cat([tpr, _interpolate(torch.tensor(fpr_range), fpr, tpr)]).sort().values
    fpr = torch.cat([fpr, torch.tensor(fpr_range)]).sort().values

    log_fpr = torch.log10(fpr)
    bounds = torch.log10(torch.tensor(fpr_range))

    lower_bound_idx = torch.where(log_fpr == bounds[0])[0]
    upper_bound_idx = torch.where(log_fpr == bounds[1])[0]

    trimmed_fpr = fpr[lower_bound_idx:upper_bound_idx+1]
    trimmed_tpr = tpr[lower_bound_idx:upper_bound_idx+1]

    # compute area and rescale it to the range of fpr
    area = _auc_compute_without_check(trimmed_fpr, trimmed_tpr) / (bounds[1] - bounds[0])
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


def _multiclass_logauc_compute(

) -> Tensor:
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

def _multilabel_logauc_compute(

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
) -> Tensor
    fpr, tpr, _ = multilabel_roc(preds, target, num_labels, thresholds, ignore_index, validate_args)
    return _multilabel_logauc_compute(fpr, tpr, fpr_range)

def logauc(

) -> Tensor:
    pass
