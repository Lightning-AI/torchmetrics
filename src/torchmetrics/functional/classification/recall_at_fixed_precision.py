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
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor

from torchmetrics.functional.classification.precision_recall_curve import (
    _binary_precision_recall_curve_arg_validation,
    _binary_precision_recall_curve_compute,
    _binary_precision_recall_curve_format,
    _binary_precision_recall_curve_tensor_validation,
    _binary_precision_recall_curve_update,
    _multiclass_precision_recall_curve_arg_validation,
    _multiclass_precision_recall_curve_compute,
    _multiclass_precision_recall_curve_format,
    _multiclass_precision_recall_curve_tensor_validation,
    _multiclass_precision_recall_curve_update,
    _multilabel_precision_recall_curve_arg_validation,
    _multilabel_precision_recall_curve_compute,
    _multilabel_precision_recall_curve_format,
    _multilabel_precision_recall_curve_tensor_validation,
    _multilabel_precision_recall_curve_update,
)


def _recall_at_precision(
    precision: Tensor,
    recall: Tensor,
    thresholds: Tensor,
    min_precision: float,
) -> Tuple[Tensor, Tensor]:
    try:
        max_recall, _, best_threshold = max(
            (r, p, t) for p, r, t in zip(precision, recall, thresholds) if p >= min_precision
        )

    except ValueError:
        max_recall = torch.tensor(0.0, device=recall.device, dtype=recall.dtype)
        best_threshold = torch.tensor(0)

    if max_recall == 0.0:
        best_threshold = torch.tensor(1e6, device=thresholds.device, dtype=thresholds.dtype)

    return max_recall, best_threshold


def _binary_recall_at_fixed_precision_arg_validation(
    min_precision: float,
    thresholds: Optional[Union[int, List[float], Tensor]] = 100,
    ignore_index: Optional[int] = None,
) -> None:
    _binary_precision_recall_curve_arg_validation(thresholds, ignore_index)
    if not isinstance(min_precision, float) and not (0 <= min_precision <= 1):
        raise ValueError(
            f"Expected argument `min_precision` to be an float in the [0,1] range, but got {min_precision}"
        )


def _binary_recall_at_fixed_precision_compute(
    state: Union[Tensor, Tuple[Tensor, Tensor]],
    thresholds: Optional[Tensor],
    min_precision: float,
    pos_label: int = 1,
) -> Tuple[Tensor, Tensor, Tensor]:
    precision, recall, thresholds = _binary_precision_recall_curve_compute(state, thresholds, pos_label)
    return _recall_at_precision(precision, recall, thresholds, min_precision)


def binary_recall_at_fixed_precision(
    preds: Tensor,
    target: Tensor,
    min_precision: float,
    thresholds: Optional[Union[int, List[float], Tensor]] = 100,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    if validate_args:
        _binary_recall_at_fixed_precision_arg_validation(min_precision, thresholds, ignore_index)
        _binary_precision_recall_curve_tensor_validation(preds, target, ignore_index)
    preds, target, thresholds = _binary_precision_recall_curve_format(preds, target, thresholds, ignore_index)
    state = _binary_precision_recall_curve_update(preds, target, thresholds)
    return _binary_recall_at_fixed_precision_compute(state, thresholds, min_precision)


def _multiclass_recall_at_fixed_precision_arg_validation(
    num_classes: int,
    min_precision: float,
    thresholds: Optional[Union[int, List[float], Tensor]] = 100,
    ignore_index: Optional[int] = None,
) -> None:
    _multiclass_precision_recall_curve_arg_validation(num_classes, thresholds, ignore_index)
    if not isinstance(min_precision, float) and not (0 <= min_precision <= 1):
        raise ValueError(
            f"Expected argument `min_precision` to be an float in the [0,1] range, but got {min_precision}"
        )


def _multiclass_recall_at_fixed_precision_arg_compute(
    state: Union[Tensor, Tuple[Tensor, Tensor]],
    num_classes: int,
    thresholds: Optional[Tensor],
    min_precision: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    precision, recall, thresholds = _multiclass_precision_recall_curve_compute(state, num_classes, thresholds)
    if isinstance(state, Tensor):
        res = [_recall_at_precision(p, r, thresholds, min_precision) for p, r in zip(precision, recall)]
        recall = torch.stack([r[0] for r in res])
        thresholds = torch.stack([r[1] for r in res])
    else:
        res = [_recall_at_precision(p, r, t, min_precision) for p, r, t in zip(precision, recall, thresholds)]
        recall = [r[0] for r in res]
        thresholds = [r[1] for r in res]
    return recall, thresholds


def multiclass_recall_at_fixed_precision(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    min_precision: float,
    thresholds: Optional[Union[int, List[float], Tensor]] = 100,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor]]]:
    if validate_args:
        _multiclass_recall_at_fixed_precision_arg_validation(num_classes, min_precision, thresholds, ignore_index)
        _multiclass_precision_recall_curve_tensor_validation(preds, target, num_classes, ignore_index)
    preds, target, thresholds = _multiclass_precision_recall_curve_format(
        preds, target, num_classes, thresholds, ignore_index
    )
    state = _multiclass_precision_recall_curve_update(preds, target, num_classes, thresholds)
    return _multiclass_recall_at_fixed_precision_arg_compute(state, num_classes, thresholds, min_precision)


def _multilabel_recall_at_fixed_precision_arg_validation(
    num_labels: int,
    min_precision: float,
    thresholds: Optional[Union[int, List[float], Tensor]] = 100,
    ignore_index: Optional[int] = None,
) -> None:
    _multilabel_precision_recall_curve_arg_validation(num_labels, thresholds, ignore_index)
    if not isinstance(min_precision, float) and not (0 <= min_precision <= 1):
        raise ValueError(
            f"Expected argument `min_precision` to be an float in the [0,1] range, but got {min_precision}"
        )


def _multilabel_recall_at_fixed_precision_arg_compute(
    state: Union[Tensor, Tuple[Tensor, Tensor]],
    num_labels: int,
    thresholds: Optional[Tensor],
    ignore_index: Optional[int],
    min_precision: float,
) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor]]]:
    precision, recall, thresholds = _multilabel_precision_recall_curve_compute(
        state, num_labels, thresholds, ignore_index
    )
    if isinstance(state, Tensor):
        res = [_recall_at_precision(p, r, thresholds, min_precision) for p, r in zip(precision, recall)]
        recall = torch.stack([r[0] for r in res])
        thresholds = torch.stack([r[1] for r in res])
    else:
        res = [_recall_at_precision(p, r, t, min_precision) for p, r, t in zip(precision, recall, thresholds)]
        recall = [r[0] for r in res]
        thresholds = [r[1] for r in res]
    return recall, thresholds


def multilabel_recall_at_fixed_precision(
    preds: Tensor,
    target: Tensor,
    num_labels: int,
    min_precision: float,
    thresholds: Optional[Union[int, List[float], Tensor]] = 100,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor]]]:
    if validate_args:
        _multilabel_recall_at_fixed_precision_arg_validation(num_labels, min_precision, thresholds, ignore_index)
        _multilabel_precision_recall_curve_tensor_validation(preds, target, num_labels, ignore_index)
    preds, target, thresholds = _multilabel_precision_recall_curve_format(
        preds, target, num_labels, thresholds, ignore_index
    )
    state = _multilabel_precision_recall_curve_update(preds, target, num_labels, thresholds)
    return _multilabel_recall_at_fixed_precision_arg_compute(state, num_labels, thresholds, ignore_index, min_precision)
