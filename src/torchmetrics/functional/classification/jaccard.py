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
from typing import Optional

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.classification.confusion_matrix import (
    _binary_confusion_matrix_arg_validation,
    _binary_confusion_matrix_format,
    _binary_confusion_matrix_tensor_validation,
    _binary_confusion_matrix_update,
    _multiclass_confusion_matrix_arg_validation,
    _multiclass_confusion_matrix_format,
    _multiclass_confusion_matrix_tensor_validation,
    _multiclass_confusion_matrix_update,
    _multilabel_confusion_matrix_arg_validation,
    _multilabel_confusion_matrix_format,
    _multilabel_confusion_matrix_tensor_validation,
    _multilabel_confusion_matrix_update,
)
from torchmetrics.utilities.compute import _safe_divide


def _jaccard_index_reduce(
    confmat: Tensor,
    average: Optional[Literal["micro", "macro", "weighted", "none", "binary"]],
) -> Tensor:
    """Perform reduction of an un-normalized confusion matrix into jaccard score.

    Args:
        confmat: tensor with un-normalized confusionmatrix
        average: reduction method

            - ``'binary'``: binary reduction, expects a 2x2 matrix
            - ``'macro'``: Calculate the metric for each class separately, and average the
              metrics across classes (with equal weights for each class).
            - ``'micro'``: Calculate the metric globally, across all samples and classes.
            - ``'weighted'``: Calculate the metric for each class separately, and average the
              metrics across classes, weighting each class by its support (``tp + fn``).
            - ``'none'`` or ``None``: Calculate the metric for each class separately, and return
              the metric for every class.
    """
    allowed_average = ["binary", "micro", "macro", "weighted", "none", None]
    if average not in allowed_average:
        raise ValueError(f"The `average` has to be one of {allowed_average}, got {average}.")
    confmat = confmat.float()
    if average == "binary":
        return confmat[1, 1] / (confmat[0, 1] + confmat[1, 0] + confmat[1, 1])
    else:
        if confmat.ndim == 3:  # multilabel
            num = confmat[:, 1, 1]
            denom = confmat[:, 1, 1] + confmat[:, 0, 1] + confmat[:, 1, 0]
        else:  # multiclass
            num = torch.diag(confmat)
            denom = confmat.sum(0) + confmat.sum(1) - num

        if average == "micro":
            num = num.sum()
            denom = denom.sum()

        jaccard = _safe_divide(num, denom)

        if average is None or average == "none":
            return jaccard
        if average == "weighted":
            weights = confmat[:, 1, 1] + confmat[:, 1, 0] if confmat.ndim == 3 else confmat.sum(1)
        else:
            weights = torch.ones_like(jaccard)
        return ((weights * jaccard) / weights.sum()).sum()


def binary_jaccard_index(
    preds: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    r"""Calculates the Jaccard index for binary tasks. The `Jaccard index`_ (also known as the intersetion over
    union or jaccard similarity coefficient) is an statistic that can be used to determine the similarity and
    diversity of a sample set. It is defined as the size of the intersection divided by the union of the sample
    sets:

    .. math:: J(A,B) = \frac{|A\cap B|}{|A\cup B|}

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Addtionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, ...)``

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        threshold: Threshold for transforming probability to binary (0,1) predictions
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        normalize: Normalization mode for confusion matrix. Choose from:

            - ``None`` or ``'none'``: no normalization (default)
            - ``'true'``: normalization over the targets (most commonly used)
            - ``'pred'``: normalization over the predictions
            - ``'all'``: normalization over the whole matrix

        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (preds is int tensor):
        >>> from torchmetrics.functional.classification import binary_jaccard_index
        >>> target = torch.tensor([1, 1, 0, 0])
        >>> preds = torch.tensor([0, 1, 0, 0])
        >>> binary_jaccard_index(preds, target)
        tensor(0.5000)

    Example (preds is float tensor):
        >>> from torchmetrics.functional.classification import binary_jaccard_index
        >>> target = torch.tensor([1, 1, 0, 0])
        >>> preds = torch.tensor([0.35, 0.85, 0.48, 0.01])
        >>> binary_jaccard_index(preds, target)
        tensor(0.5000)
    """
    if validate_args:
        _binary_confusion_matrix_arg_validation(threshold, ignore_index)
        _binary_confusion_matrix_tensor_validation(preds, target, ignore_index)
    preds, target = _binary_confusion_matrix_format(preds, target, threshold, ignore_index)
    confmat = _binary_confusion_matrix_update(preds, target)
    return _jaccard_index_reduce(confmat, average="binary")


def _multiclass_jaccard_index_arg_validation(
    num_classes: int,
    ignore_index: Optional[int] = None,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = None,
) -> None:
    _multiclass_confusion_matrix_arg_validation(num_classes, ignore_index)
    allowed_average = ("micro", "macro", "weighted", "none", None)
    if average not in allowed_average:
        raise ValueError(f"Expected argument `average` to be one of {allowed_average}, but got {average}.")


def multiclass_jaccard_index(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    r"""Calculates the Jaccard index for multiclass tasks. The `Jaccard index`_ (also known as the intersetion over
    union or jaccard similarity coefficient) is an statistic that can be used to determine the similarity and
    diversity of a sample set. It is defined as the size of the intersection divided by the union of the sample
    sets:

    .. math:: J(A,B) = \frac{|A\cap B|}{|A\cup B|}

    Accepts the following input tensors:

    - ``preds``: ``(N, ...)`` (int tensor) or ``(N, C, ..)`` (float tensor). If preds is a floating point
      we apply ``torch.argmax`` along the ``C`` dimension to automatically convert probabilities/logits into
      an int tensor.
    - ``target`` (int tensor): ``(N, ...)``

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        num_classes: Integer specifing the number of classes
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: Calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: Calculates statistic for each label and applies no reduction

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (pred is integer tensor):
        >>> from torchmetrics.functional.classification import multiclass_jaccard_index
        >>> target = torch.tensor([2, 1, 0, 0])
        >>> preds = torch.tensor([2, 1, 0, 1])
        >>> multiclass_jaccard_index(preds, target, num_classes=3)
        tensor(0.6667)

    Example (pred is float tensor):
        >>> from torchmetrics.functional.classification import multiclass_jaccard_index
        >>> target = torch.tensor([2, 1, 0, 0])
        >>> preds = torch.tensor([
        ...   [0.16, 0.26, 0.58],
        ...   [0.22, 0.61, 0.17],
        ...   [0.71, 0.09, 0.20],
        ...   [0.05, 0.82, 0.13],
        ... ])
        >>> multiclass_jaccard_index(preds, target, num_classes=3)
        tensor(0.6667)
    """
    if validate_args:
        _multiclass_jaccard_index_arg_validation(num_classes, ignore_index, average)
        _multiclass_confusion_matrix_tensor_validation(preds, target, num_classes, ignore_index)
    preds, target = _multiclass_confusion_matrix_format(preds, target, ignore_index)
    confmat = _multiclass_confusion_matrix_update(preds, target, num_classes)
    return _jaccard_index_reduce(confmat, average=average)


def _multilabel_jaccard_index_arg_validation(
    num_labels: int,
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
) -> None:
    _multilabel_confusion_matrix_arg_validation(num_labels, threshold, ignore_index)
    allowed_average = ("micro", "macro", "weighted", "none", None)
    if average not in allowed_average:
        raise ValueError(f"Expected argument `average` to be one of {allowed_average}, but got {average}.")


def multilabel_jaccard_index(
    preds: Tensor,
    target: Tensor,
    num_labels: int,
    threshold: float = 0.5,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    r"""Calculates the Jaccard index for multilabel tasks. The `Jaccard index`_ (also known as the intersetion over
    union or jaccard similarity coefficient) is an statistic that can be used to determine the similarity and
    diversity of a sample set. It is defined as the size of the intersection divided by the union of the sample
    sets:

    .. math:: J(A,B) = \frac{|A\cap B|}{|A\cup B|}

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, C, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Addtionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, C, ...)``

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        num_classes: Integer specifing the number of labels
        threshold: Threshold for transforming probability to binary (0,1) predictions
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: Calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: Calculates statistic for each label and applies no reduction

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (preds is int tensor):
        >>> from torchmetrics.functional.classification import multilabel_jaccard_index
        >>> target = torch.tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = torch.tensor([[0, 0, 1], [1, 0, 1]])
        >>> multilabel_jaccard_index(preds, target, num_labels=3)
        tensor(0.5000)

    Example (preds is float tensor):
        >>> from torchmetrics.functional.classification import multilabel_jaccard_index
        >>> target = torch.tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = torch.tensor([[0.11, 0.22, 0.84], [0.73, 0.33, 0.92]])
        >>> multilabel_jaccard_index(preds, target, num_labels=3)
        tensor(0.5000)
    """
    if validate_args:
        _multilabel_jaccard_index_arg_validation(num_labels, threshold, ignore_index)
        _multilabel_confusion_matrix_tensor_validation(preds, target, num_labels, ignore_index)
    preds, target = _multilabel_confusion_matrix_format(preds, target, num_labels, threshold, ignore_index)
    confmat = _multilabel_confusion_matrix_update(preds, target, num_labels)
    return _jaccard_index_reduce(confmat, average=average)


def jaccard_index(
    preds: Tensor,
    target: Tensor,
    task: Literal["binary", "multiclass", "multilabel"],
    threshold: float = 0.5,
    num_classes: Optional[int] = None,
    num_labels: Optional[int] = None,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    r"""Calculates the Jaccard index. The `Jaccard index`_ (also known as the intersetion over
    union or jaccard similarity coefficient) is an statistic that can be used to determine the similarity and
    diversity of a sample set. It is defined as the size of the intersection divided by the union of the sample
    sets:

    .. math:: J(A,B) = \frac{|A\cap B|}{|A\cup B|}

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``multilabel``. See the documentation of
    :func:`binary_jaccard_index`, :func:`multiclass_jaccard_index` and :func:`multilabel_jaccard_index` for
    the specific details of each argument influence and examples.

    Legacy Example:
        >>> target = torch.randint(0, 2, (10, 25, 25))
        >>> pred = torch.tensor(target)
        >>> pred[2:5, 7:13, 9:15] = 1 - pred[2:5, 7:13, 9:15]
        >>> jaccard_index(pred, target, task="multiclass", num_classes=2)
        tensor(0.9660)
    """
    if task == "binary":
        return binary_jaccard_index(preds, target, threshold, ignore_index, validate_args)
    if task == "multiclass":
        assert isinstance(num_classes, int)
        return multiclass_jaccard_index(preds, target, num_classes, average, ignore_index, validate_args)
    if task == "multilabel":
        assert isinstance(num_labels, int)
        return multilabel_jaccard_index(preds, target, num_labels, threshold, average, ignore_index, validate_args)
    raise ValueError(
        f"Expected argument `task` to either be `'binary'`, `'multiclass'` or `'multilabel'` but got {task}"
    )
