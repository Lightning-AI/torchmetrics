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

from torchmetrics.functional.classification.stat_scores import _del_column
from torchmetrics.utilities.checks import _input_format_classification
from torchmetrics.utilities.enums import DataType


def _accuracy_update(
    preds: Tensor,
    target: Tensor,
    threshold: float,
    top_k: Optional[int],
    ignore_index: Optional[int],
    subset_accuracy: bool,
) -> Tuple[Tensor, Tensor]:

    preds, target, mode = _input_format_classification(preds, target, threshold=threshold, top_k=top_k)
    correct, total = None, None

    # Delete what is in ignore_index, if applicable (and classes don't matter):
    if ignore_index is not None:
        preds = _del_column(preds, ignore_index)
        target = _del_column(target, ignore_index)

    if mode == DataType.BINARY or (mode == DataType.MULTILABEL and subset_accuracy):
        correct = (preds == target).all(dim=1).sum()
        total = tensor(target.shape[0], device=target.device)
    elif mode == DataType.MULTILABEL and not subset_accuracy:
        correct = (preds == target).sum()
        total = tensor(target.numel(), device=target.device)
    elif mode == DataType.MULTICLASS or (mode == DataType.MULTIDIM_MULTICLASS and not subset_accuracy):
        correct = (preds * target).sum()
        total = target.sum()
    elif mode == DataType.MULTIDIM_MULTICLASS and subset_accuracy:
        sample_correct = (preds * target).sum(dim=(1, 2))
        correct = (sample_correct == target.shape[2]).sum()
        total = tensor(target.shape[0], device=target.device)

    return correct, total


def _accuracy_compute(correct: Tensor, total: Tensor) -> Tensor:
    return correct.float() / total


def accuracy(
    preds: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    ignore_index: Optional[int] = None,
    subset_accuracy: bool = False,
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
        ignore_index:
            Integer specifying a target class to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method. If an index is ignored, and ``average=None``
            or ``'none'``, the score for the ignored class will be returned as ``nan``.
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

    correct, total = _accuracy_update(preds, target, threshold, top_k, ignore_index, subset_accuracy)
    return _accuracy_compute(correct, total)
