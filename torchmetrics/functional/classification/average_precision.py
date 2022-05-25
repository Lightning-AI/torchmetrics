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
from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from torchmetrics.functional.classification.precision_recall_curve import (
    _precision_recall_curve_compute,
    _precision_recall_curve_update,
)
from torchmetrics.utilities.data import _bincount


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
    if average == "micro":
        if preds.ndim == target.ndim:
            # Considering each element of the label indicator matrix as a label
            preds = preds.flatten()
            target = target.flatten()
            num_classes = 1
        else:
            raise ValueError("Cannot use `micro` average with multi-class input")

    return preds, target, num_classes, pos_label


def _average_precision_compute(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    pos_label: Optional[int] = None,
    average: Optional[str] = "macro",
    sample_weights: Optional[Sequence] = None,
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
        sample_weights: sample weights for each data point

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

    # todo: `sample_weights` is unused
    precision, recall, _ = _precision_recall_curve_compute(preds, target, num_classes, pos_label)
    if average == "weighted":
        if preds.ndim == target.ndim and target.ndim > 1:
            weights = target.sum(dim=0).float()
        else:
            weights = _bincount(target, minlength=num_classes).float()
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
    if average is None:
        return res
    allowed_average = ("micro", "macro", "weighted", None)
    raise ValueError(f"Expected argument `average` to be one of {allowed_average}" f" but got {average}")


def average_precision(
    preds: Tensor,
    target: Tensor,
    num_classes: Optional[int] = None,
    pos_label: Optional[int] = None,
    average: Optional[str] = "macro",
    sample_weights: Optional[Sequence] = None,
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

        sample_weights: sample weights for each data point

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
    # fixme: `sample_weights` is unused
    preds, target, num_classes, pos_label = _average_precision_update(preds, target, num_classes, pos_label, average)
    return _average_precision_compute(preds, target, num_classes, pos_label, average, sample_weights)
