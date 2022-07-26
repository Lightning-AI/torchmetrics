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
import warnings
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from typing_extensions import Literal

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
    _precision_recall_curve_compute,
    _precision_recall_curve_update,
)
from torchmetrics.utilities.data import _bincount
from torchmetrics.utilities.prints import rank_zero_warn


def _reduce_average_precision(
    precision: Union[Tensor, List[Tensor]],
    recall: Union[Tensor, List[Tensor]],
    average: Optional[Literal["macro", "weighted", "none"]] = "macro",
    weights: Optional[Tensor] = None,
) -> Tensor:
    """Utility function for reducing multiple average precision score into one number."""
    res = []
    if isinstance(precision, Tensor):
        res = -torch.sum((recall[:, 1:] - recall[:, :-1]) * precision[:, :-1], 1)
    else:
        for p, r in zip(precision, recall):
            res.append(-torch.sum((r[1:] - r[:-1]) * p[:-1]))
        res = torch.stack(res)
    if average is None or average == "none":
        return res
    if torch.isnan(res).any():
        rank_zero_warn(
            f"Average precision score for one or more classes was `nan`. Ignoring these classes in {average}-average",
            UserWarning,
        )
    if average == "macro":
        return res[~torch.isnan(res)].mean()

    weights = weights / weights.sum()
    return (res * weights)[~torch.isnan(res)].sum()


def _binary_average_precision_compute(
    state: Union[Tensor, Tuple[Tensor, Tensor]],
    thresholds: Optional[Tensor],
) -> Tensor:
    precision, recall, _ = _binary_precision_recall_curve_compute(state, thresholds)
    return -torch.sum((recall[1:] - recall[:-1]) * precision[:-1])


def binary_average_precision(
    preds: Tensor,
    target: Tensor,
    thresholds: Optional[Union[int, List[float], Tensor]] = 100,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    if validate_args:
        _binary_precision_recall_curve_arg_validation(thresholds, ignore_index)
        _binary_precision_recall_curve_tensor_validation(preds, target, ignore_index)
    preds, target, thresholds = _binary_precision_recall_curve_format(preds, target, thresholds, ignore_index)
    state = _binary_precision_recall_curve_update(preds, target, thresholds)
    return _binary_average_precision_compute(state, thresholds)


def _multiclass_average_precision_arg_validation(
    num_classes: int,
    average: Optional[Literal["macro", "weighted", "none"]] = "macro",
    thresholds: Optional[Union[int, List[float], Tensor]] = 100,
    ignore_index: Optional[int] = None,
) -> None:
    _multiclass_precision_recall_curve_arg_validation(num_classes, thresholds, ignore_index)
    allowed_average = ("macro", "weighted", "none", None)
    if average not in allowed_average:
        raise ValueError(f"Expected argument `average` to be one of {allowed_average} but got {average}")


def _multiclass_average_precision_compute(
    state: Union[Tensor, Tuple[Tensor, Tensor]],
    num_classes: int,
    average: Optional[Literal["macro", "weighted", "none"]] = "macro",
    thresholds: Optional[Tensor] = None,
) -> Tensor:
    precision, recall, _ = _multiclass_precision_recall_curve_compute(state, num_classes, thresholds)
    return _reduce_average_precision(
        precision,
        recall,
        average,
        weights=_bincount(state[1], minlength=num_classes).float() if thresholds is None else state[0][:, 1, :].sum(-1),
    )


def multiclass_average_precision(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    average: Optional[Literal["macro", "weighted", "none"]] = "macro",
    thresholds: Optional[Union[int, List[float], Tensor]] = 100,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    if validate_args:
        _multiclass_average_precision_arg_validation(num_classes, average, thresholds, ignore_index)
        _multiclass_precision_recall_curve_tensor_validation(preds, target, num_classes, ignore_index)
    preds, target, thresholds = _multiclass_precision_recall_curve_format(
        preds, target, num_classes, thresholds, ignore_index
    )
    state = _multiclass_precision_recall_curve_update(preds, target, num_classes, thresholds)
    return _multiclass_average_precision_compute(state, num_classes, average, thresholds)


def _multilabel_average_precision_arg_validation(
    num_labels: int,
    average: Optional[Literal["micro", "macro", "weighted", "none"]],
    thresholds: Optional[Union[int, List[float], Tensor]] = 100,
    ignore_index: Optional[int] = None,
) -> None:
    _multilabel_precision_recall_curve_arg_validation(num_labels, thresholds, ignore_index)
    allowed_average = ("micro", "macro", "weighted", "none", None)
    if average not in allowed_average:
        raise ValueError(f"Expected argument `average` to be one of {allowed_average} but got {average}")


def _multilabel_average_precision_compute(
    state: Union[Tensor, Tuple[Tensor, Tensor]],
    num_labels: int,
    average: Optional[Literal["micro", "macro", "weighted", "none"]],
    thresholds: Optional[Tensor],
    ignore_index: Optional[int] = None,
) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor]]]:
    if average == "micro":
        return _binary_average_precision_compute(
            [state[0].flatten(), state[1].flatten()] if thresholds is None else state.sum(1),
            thresholds,
        )
    else:
        precision, recall, _ = _multilabel_precision_recall_curve_compute(state, num_labels, thresholds)
        return _reduce_average_precision(
            precision,
            recall,
            average,
            weights=(state[1] == 1).sum(dim=0).float() if thresholds is None else state[0][:, 1, :].sum(-1),
        )


def multilabel_average_precision(
    preds: Tensor,
    target: Tensor,
    num_labels: int,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
    thresholds: Optional[Union[int, List[float], Tensor]] = 100,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    if validate_args:
        _multilabel_average_precision_arg_validation(num_labels, average, thresholds, ignore_index)
        _multilabel_precision_recall_curve_tensor_validation(preds, target, num_labels, ignore_index)
    preds, target, thresholds = _multilabel_precision_recall_curve_format(
        preds, target, num_labels, thresholds, ignore_index
    )
    state = _multilabel_precision_recall_curve_update(preds, target, num_labels, thresholds)
    return _multilabel_average_precision_compute(state, num_labels, average, thresholds, ignore_index)


# -------------------------- Old stuff --------------------------


def _average_precision_update(
    preds: Tensor,
    target: Tensor,
    num_classes: Optional[int] = None,
    pos_label: Optional[int] = None,
    average: Optional[str] = "macro",
) -> Tuple[Tensor, Tensor, int, Optional[int]]:
    """Format the predictions and target based on the ``num_classes``, ``pos_label`` and ``average`` parameter.

    Args:
        preds: predictions from model (logits or probabilities)
        target: ground truth values
        num_classes: integer with number of classes.
        pos_label: integer determining the positive class. Default is ``None`` which for binary problem is translated
            to 1. For multiclass problems this argument should not be set as we iteratively change it in the
            range ``[0, num_classes-1]``
        average: reduction method for multi-class or multi-label problems
    """
    preds, target, num_classes, pos_label = _precision_recall_curve_update(preds, target, num_classes, pos_label)
    if average == "micro" and preds.ndim != target.ndim:
        raise ValueError("Cannot use `micro` average with multi-class input")

    return preds, target, num_classes, pos_label


def _average_precision_compute(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    pos_label: Optional[int] = None,
    average: Optional[str] = "macro",
) -> Union[List[Tensor], Tensor]:
    """Computes the average precision score.

    Args:
        preds: predictions from model (logits or probabilities)
        target: ground truth values
        num_classes: integer with number of classes.
        pos_label: integer determining the positive class. Default is ``None`` which for binary problem is translated
            to 1. For multiclass problems his argument should not be set as we iteratively change it in the
            range ``[0, num_classes-1]``
        average: reduction method for multi-class or multi-label problems

    Example:
        >>> # binary case
        >>> preds = torch.tensor([0, 1, 2, 3])
        >>> target = torch.tensor([0, 1, 1, 1])
        >>> pos_label = 1
        >>> preds, target, num_classes, pos_label = _average_precision_update(preds, target, pos_label=pos_label)
        >>> _average_precision_compute(preds, target, num_classes, pos_label)
        tensor(1.)

        >>> # multiclass case
        >>> preds = torch.tensor([[0.75, 0.05, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.75, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.75, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.05, 0.75, 0.05]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> num_classes = 5
        >>> preds, target, num_classes, pos_label = _average_precision_update(preds, target, num_classes)
        >>> _average_precision_compute(preds, target, num_classes, average=None)
        [tensor(1.), tensor(1.), tensor(0.2500), tensor(0.2500), tensor(nan)]
    """

    if average == "micro" and preds.ndim == target.ndim:
        preds = preds.flatten()
        target = target.flatten()
        num_classes = 1

    precision, recall, _ = _precision_recall_curve_compute(preds, target, num_classes, pos_label)
    if average == "weighted":
        if preds.ndim == target.ndim and target.ndim > 1:
            weights = target.sum(dim=0).float()
        else:
            weights = _bincount(target, minlength=max(num_classes, 2)).float()
        weights = weights / torch.sum(weights)
    else:
        weights = None
    return _average_precision_compute_with_precision_recall(precision, recall, num_classes, average, weights)


def _average_precision_compute_with_precision_recall(
    precision: Tensor,
    recall: Tensor,
    num_classes: int,
    average: Optional[str] = "macro",
    weights: Optional[Tensor] = None,
) -> Union[List[Tensor], Tensor]:
    """Computes the average precision score from precision and recall.

    Args:
        precision: precision values
        recall: recall values
        num_classes: integer with number of classes. Not nessesary to provide
            for binary problems.
        average: reduction method for multi-class or multi-label problems
        weights: weights to use when average='weighted'

    Example:
        >>> # binary case
        >>> preds = torch.tensor([0, 1, 2, 3])
        >>> target = torch.tensor([0, 1, 1, 1])
        >>> pos_label = 1
        >>> preds, target, num_classes, pos_label = _average_precision_update(preds, target, pos_label=pos_label)
        >>> precision, recall, _ = _precision_recall_curve_compute(preds, target, num_classes, pos_label)
        >>> _average_precision_compute_with_precision_recall(precision, recall, num_classes, average=None)
        tensor(1.)

        >>> # multiclass case
        >>> preds = torch.tensor([[0.75, 0.05, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.75, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.75, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.05, 0.75, 0.05]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> num_classes = 5
        >>> preds, target, num_classes, pos_label = _average_precision_update(preds, target, num_classes)
        >>> precision, recall, _ = _precision_recall_curve_compute(preds, target, num_classes)
        >>> _average_precision_compute_with_precision_recall(precision, recall, num_classes, average=None)
        [tensor(1.), tensor(1.), tensor(0.2500), tensor(0.2500), tensor(nan)]
    """

    # Return the step function integral
    # The following works because the last entry of precision is
    # guaranteed to be 1, as returned by precision_recall_curve
    if num_classes == 1:
        return -torch.sum((recall[1:] - recall[:-1]) * precision[:-1])

    res = []
    for p, r in zip(precision, recall):
        res.append(-torch.sum((r[1:] - r[:-1]) * p[:-1]))

    # Reduce
    if average in ("macro", "weighted"):
        res = torch.stack(res)
        if torch.isnan(res).any():
            warnings.warn(
                "Average precision score for one or more classes was `nan`. Ignoring these classes in average",
                UserWarning,
            )
        if average == "macro":
            return res[~torch.isnan(res)].mean()
        weights = torch.ones_like(res) if weights is None else weights
        return (res * weights)[~torch.isnan(res)].sum()
    if average is None or average == "none":
        return res
    allowed_average = ("micro", "macro", "weighted", "none", None)
    raise ValueError(f"Expected argument `average` to be one of {allowed_average}" f" but got {average}")


def average_precision(
    preds: Tensor,
    target: Tensor,
    num_classes: Optional[int] = None,
    pos_label: Optional[int] = None,
    average: Optional[str] = "macro",
) -> Union[List[Tensor], Tensor]:
    """Computes the average precision score.

    Args:
        preds: predictions from model (logits or probabilities)
        target: ground truth values
        num_classes: integer with number of classes. Not nessesary to provide
            for binary problems.
        pos_label: integer determining the positive class. Default is ``None`` which for binary problem is translated
            to 1. For multiclass problems his argument should not be set as we iteratively change it in the
            range ``[0, num_classes-1]``
        average:
            defines the reduction that is applied in the case of multiclass and multilabel input.
            Should be one of the following:

            - ``'macro'`` [default]: Calculate the metric for each class separately, and average the
              metrics across classes (with equal weights for each class).
            - ``'micro'``: Calculate the metric globally, across all samples and classes. Cannot be
              used with multiclass input.
            - ``'weighted'``: Calculate the metric for each class separately, and average the
              metrics across classes, weighting each class by its support.
            - ``'none'`` or ``None``: Calculate the metric for each class separately, and return
              the metric for every class.

    Returns:
        tensor with average precision. If multiclass will return list
        of such tensors, one for each class

    Example (binary case):
        >>> from torchmetrics.functional import average_precision
        >>> pred = torch.tensor([0, 1, 2, 3])
        >>> target = torch.tensor([0, 1, 1, 1])
        >>> average_precision(pred, target, pos_label=1)
        tensor(1.)

    Example (multiclass case):
        >>> pred = torch.tensor([[0.75, 0.05, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.75, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.75, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.05, 0.75, 0.05]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> average_precision(pred, target, num_classes=5, average=None)
        [tensor(1.), tensor(1.), tensor(0.2500), tensor(0.2500), tensor(nan)]
    """
    preds, target, num_classes, pos_label = _average_precision_update(preds, target, num_classes, pos_label, average)
    return _average_precision_compute(preds, target, num_classes, pos_label, average)
