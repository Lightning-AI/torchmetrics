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
from torch import Tensor, tensor

from torchmetrics.utilities.checks import _input_format_classification, _check_classification_inputs
from torchmetrics.utilities.enums import DataType
from torchmetrics.functional.classification.stat_scores import _stat_scores_update
from torchmetrics.classification.stat_scores import _reduce_stat_scores


def _check_subset_validity(mode, preds, target):
    return (
        mode == DataType.MULTILABEL
        or mode == DataType.MULTIDIM_MULTICLASS
    )


def _mode(
    preds: Tensor,
    target: Tensor,
    threshold: float,
    top_k: Optional[int],
    num_classes: Optional[int],
    is_multiclass: Optional[bool]
) -> DataType:
    mode = _check_classification_inputs(
        preds, target, threshold=threshold, top_k=top_k, num_classes=num_classes, multiclass=is_multiclass
    )
    return mode


def _accuracy_update(
    preds: Tensor,
    target: Tensor,
    reduce: str,
    mdmc_reduce: str,
    threshold: float,
    num_classes: Optional[int],
    top_k: Optional[int],
    is_multiclass: Optional[bool],
    ignore_index: Optional[int],
    mode: DataType
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    if mode == DataType.MULTILABEL and top_k:
        raise ValueError("You can not use the `top_k` parameter to calculate accuracy for multi-label inputs.")

    tp, fp, tn, fn = _stat_scores_update(
        preds,
        target,
        reduce=reduce,
        mdmc_reduce=mdmc_reduce,
        threshold=threshold,
        num_classes=num_classes,
        top_k=top_k,
        is_multiclass=is_multiclass,
        ignore_index=ignore_index,
    )
    return tp, fp, tn, fn


def _accuracy_compute(
    tp: Tensor, fp: Tensor, tn: Tensor, fn: Tensor, average: str, mdmc_average: str, mode: DataType
) -> Tensor:
    simple_average = ["micro", "samples"]
    if (mode == DataType.BINARY and average in simple_average) or mode == DataType.MULTILABEL:
        numerator = tp + tn
        denominator = tp + tn + fp + fn
    else:
        numerator = tp
        denominator = tp + fn
    return _reduce_stat_scores(
        numerator=numerator,
        denominator=denominator,
        weights=None if average != "weighted" else tp + fn,
        average=average,
        mdmc_average=mdmc_average,
    )


def _subset_accuracy_update(
    preds: Tensor, target: Tensor, threshold: float, top_k: Optional[int],
) -> Tuple[Tensor, Tensor]:

    preds, target, mode = _input_format_classification(preds, target, threshold=threshold, top_k=top_k)

    if mode == DataType.MULTILABEL and top_k:
        raise ValueError("You can not use the `top_k` parameter to calculate accuracy for multi-label inputs.")

    if mode == DataType.MULTILABEL:
        correct = (preds == target).all(dim=1).sum()
        total = tensor(target.shape[0], device=target.device)
    elif mode == DataType.MULTICLASS:
        correct = (preds * target).sum()
        total = target.sum()
    elif mode == DataType.MULTIDIM_MULTICLASS:
        sample_correct = (preds * target).sum(dim=(1, 2))
        correct = (sample_correct == target.shape[2]).sum()
        total = tensor(target.shape[0], device=target.device)

    return correct, total


def _subset_accuracy_compute(correct: Tensor, total: Tensor) -> Tensor:
    return correct.float() / total


def accuracy(
    preds: Tensor,
    target: Tensor,
    average: str = "micro",
    mdmc_average: Optional[str] = "global",
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    subset_accuracy: bool = False,
    num_classes: Optional[int] = None,
    is_multiclass: Optional[bool] = None,
    ignore_index: Optional[int] = None,
) -> Tensor:
    r"""Computes `Accuracy <https://en.wikipedia.org/wiki/Accuracy_and_precision>`_:

    .. math::
        \text{Accuracy} = \frac{1}{N}\sum_i^N 1(y_i = \hat{y}_i)

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a
    tensor of predictions.

    For multi-class and multi-dimensional multi-class data with probability predictions, the
    parameter ``top_k`` generalizes this metric to a Top-K accuracy metric: for each sample the
    top-K highest probability items are considered to find the correct label.

    For multi-label and multi-dimensional multi-class inputs, this metric computes the "global"
    accuracy by default, which counts all labels or sub-samples separately. This can be
    changed to subset accuracy (which requires all labels or sub-samples in the sample to
    be correctly predicted) by setting ``subset_accuracy=True``.

    Accepts all input types listed in :ref:`references/modules:input types`.

    Args:
        preds: Predictions from model (probabilities, or labels)
        target: Ground truth labels
        threshold:
            Threshold probability value for transforming probability predictions to binary
            (0,1) predictions, in the case of binary or multi-label inputs.
        top_k:
            Number of highest probability predictions considered to find the correct label, relevant
            only for (multi-dimensional) multi-class inputs with probability predictions. The
            default value (``None``) will be interpreted as 1 for these inputs.

            Should be left at default (``None``) for all other types of inputs.
        subset_accuracy:
            Whether to compute subset accuracy for multi-label and multi-dimensional
            multi-class inputs (has no effect for other input types).

            - For multi-label inputs, if the parameter is set to ``True``, then all labels for
              each sample must be correctly predicted for the sample to count as correct. If it
              is set to ``False``, then all labels are counted separately - this is equivalent to
              flattening inputs beforehand (i.e. ``preds = preds.flatten()`` and same for ``target``).

            - For multi-dimensional multi-class inputs, if the parameter is set to ``True``, then all
              sub-sample (on the extra axis) must be correct for the sample to be counted as correct.
              If it is set to ``False``, then all sub-samples are counter separately - this is equivalent,
              in the case of label predictions, to flattening the inputs beforehand (i.e.
              ``preds = preds.flatten()`` and same for ``target``). Note that the ``top_k`` parameter
              still applies in both cases, if set.

    Raises:
        ValueError:
            If ``top_k`` parameter is set for ``multi-label`` inputs.

    Example:
        >>> from torchmetrics.functional import accuracy
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> preds = torch.tensor([0, 2, 1, 3])
        >>> accuracy(preds, target)
        tensor(0.5000)

        >>> target = torch.tensor([0, 1, 2])
        >>> preds = torch.tensor([[0.1, 0.9, 0], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3]])
        >>> accuracy(preds, target, top_k=2)
        tensor(0.6667)
    """

    mode = _mode(preds, target, threshold, top_k, num_classes, is_multiclass)

    if subset_accuracy and _check_subset_validity(mode, preds, target):
        correct, total = _subset_accuracy_update(preds, target, threshold, top_k)
        return _subset_accuracy_compute(correct, total)
    else:
        tp, fp, tn, fn = _accuracy_update(
            preds, target, average, mdmc_average, threshold, num_classes, top_k, is_multiclass, ignore_index, mode
        )
        return _accuracy_compute(tp, fp, tn, fn, average, mdmc_average, mode)
