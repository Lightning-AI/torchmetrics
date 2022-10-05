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
    _confusion_matrix_update,
    _multiclass_confusion_matrix_arg_validation,
    _multiclass_confusion_matrix_format,
    _multiclass_confusion_matrix_tensor_validation,
    _multiclass_confusion_matrix_update,
    _multilabel_confusion_matrix_arg_validation,
    _multilabel_confusion_matrix_format,
    _multilabel_confusion_matrix_tensor_validation,
    _multilabel_confusion_matrix_update,
)
from torchmetrics.utilities.prints import rank_zero_warn


def _matthews_corrcoef_reduce(confmat: Tensor) -> Tensor:
    """Reduce an un-normalized confusion matrix of shape (n_classes, n_classes) into the matthews corrcoef
    score."""
    # convert multilabel into binary
    confmat = confmat.sum(0) if confmat.ndim == 3 else confmat

    tk = confmat.sum(dim=-1).float()
    pk = confmat.sum(dim=-2).float()
    c = torch.trace(confmat).float()
    s = confmat.sum().float()

    cov_ytyp = c * s - sum(tk * pk)
    cov_ypyp = s**2 - sum(pk * pk)
    cov_ytyt = s**2 - sum(tk * tk)

    denom = cov_ypyp * cov_ytyt
    if denom == 0:
        return torch.tensor(0, dtype=confmat.dtype, device=confmat.device)
    else:
        return cov_ytyp / torch.sqrt(denom)


def binary_matthews_corrcoef(
    preds: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    r"""Calculates `Matthews correlation coefficient`_ for binary tasks. This metric measures the general
    correlation or quality of a classification.

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
        >>> from torchmetrics.functional.classification import binary_matthews_corrcoef
        >>> target = torch.tensor([1, 1, 0, 0])
        >>> preds = torch.tensor([0, 1, 0, 0])
        >>> binary_matthews_corrcoef(preds, target)
        tensor(0.5774)

    Example (preds is float tensor):
        >>> from torchmetrics.functional.classification import binary_matthews_corrcoef
        >>> target = torch.tensor([1, 1, 0, 0])
        >>> preds = torch.tensor([0.35, 0.85, 0.48, 0.01])
        >>> binary_matthews_corrcoef(preds, target)
        tensor(0.5774)
    """
    if validate_args:
        _binary_confusion_matrix_arg_validation(threshold, ignore_index, normalize=None)
        _binary_confusion_matrix_tensor_validation(preds, target, ignore_index)
    preds, target = _binary_confusion_matrix_format(preds, target, threshold, ignore_index)
    confmat = _binary_confusion_matrix_update(preds, target)
    return _matthews_corrcoef_reduce(confmat)


def multiclass_matthews_corrcoef(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    r"""Calculates `Matthews correlation coefficient`_ for multiclass tasks. This metric measures the general
    correlation or quality of a classification.

    Accepts the following input tensors:

    - ``preds``: ``(N, ...)`` (int tensor) or ``(N, C, ..)`` (float tensor). If preds is a floating point
      we apply ``torch.argmax`` along the ``C`` dimension to automatically convert probabilities/logits into
      an int tensor.
    - ``target`` (int tensor): ``(N, ...)``

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        num_classes: Integer specifing the number of classes
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

    Example (pred is integer tensor):
        >>> from torchmetrics.functional.classification import multiclass_matthews_corrcoef
        >>> target = torch.tensor([2, 1, 0, 0])
        >>> preds = torch.tensor([2, 1, 0, 1])
        >>> multiclass_matthews_corrcoef(preds, target, num_classes=3)
        tensor(0.7000)

    Example (pred is float tensor):
        >>> from torchmetrics.functional.classification import multiclass_matthews_corrcoef
        >>> target = torch.tensor([2, 1, 0, 0])
        >>> preds = torch.tensor([
        ...   [0.16, 0.26, 0.58],
        ...   [0.22, 0.61, 0.17],
        ...   [0.71, 0.09, 0.20],
        ...   [0.05, 0.82, 0.13],
        ... ])
        >>> multiclass_matthews_corrcoef(preds, target, num_classes=3)
        tensor(0.7000)
    """
    if validate_args:
        _multiclass_confusion_matrix_arg_validation(num_classes, ignore_index, normalize=None)
        _multiclass_confusion_matrix_tensor_validation(preds, target, num_classes, ignore_index)
    preds, target = _multiclass_confusion_matrix_format(preds, target, ignore_index)
    confmat = _multiclass_confusion_matrix_update(preds, target, num_classes)
    return _matthews_corrcoef_reduce(confmat)


def multilabel_matthews_corrcoef(
    preds: Tensor,
    target: Tensor,
    num_labels: int,
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    r"""Calculates `Matthews correlation coefficient`_ for multilabel tasks. This metric measures the general
    correlation or quality of a classification.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, C, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Addtionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, C, ...)``

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        num_classes: Integer specifing the number of labels
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
        >>> from torchmetrics.functional.classification import multilabel_matthews_corrcoef
        >>> target = torch.tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = torch.tensor([[0, 0, 1], [1, 0, 1]])
        >>> multilabel_matthews_corrcoef(preds, target, num_labels=3)
        tensor(0.3333)

    Example (preds is float tensor):
        >>> from torchmetrics.functional.classification import multilabel_matthews_corrcoef
        >>> target = torch.tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = torch.tensor([[0.11, 0.22, 0.84], [0.73, 0.33, 0.92]])
        >>> multilabel_matthews_corrcoef(preds, target, num_labels=3)
        tensor(0.3333)
    """
    if validate_args:
        _multilabel_confusion_matrix_arg_validation(num_labels, threshold, ignore_index, normalize=None)
        _multilabel_confusion_matrix_tensor_validation(preds, target, num_labels, ignore_index)
    preds, target = _multilabel_confusion_matrix_format(preds, target, num_labels, threshold, ignore_index)
    confmat = _multilabel_confusion_matrix_update(preds, target, num_labels)
    return _matthews_corrcoef_reduce(confmat)


_matthews_corrcoef_update = _confusion_matrix_update


def _matthews_corrcoef_compute(confmat: Tensor) -> Tensor:
    """Computes Matthews correlation coefficient.

    Args:
        confmat: Confusion matrix

    Example:
        >>> target = torch.tensor([1, 1, 0, 0])
        >>> preds = torch.tensor([0, 1, 0, 0])
        >>> confmat = _matthews_corrcoef_update(preds, target, num_classes=2)
        >>> _matthews_corrcoef_compute(confmat)
        tensor(0.5774)
    """

    tk = confmat.sum(dim=1).float()
    pk = confmat.sum(dim=0).float()
    c = torch.trace(confmat).float()
    s = confmat.sum().float()

    cov_ytyp = c * s - sum(tk * pk)
    cov_ypyp = s**2 - sum(pk * pk)
    cov_ytyt = s**2 - sum(tk * tk)

    if cov_ypyp * cov_ytyt == 0:
        return torch.tensor(0, dtype=confmat.dtype, device=confmat.device)
    else:
        return cov_ytyp / torch.sqrt(cov_ytyt * cov_ypyp)


def matthews_corrcoef(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    threshold: float = 0.5,
    task: Optional[Literal["binary", "multiclass", "multilabel"]] = None,
    num_labels: Optional[int] = None,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    r"""Matthews correlation coefficient.

    .. note::
        From v0.10 an ``'binary_*'``, ``'multiclass_*'``, ``'multilabel_*'`` version now exist of each classification
        metric. Moving forward we recommend using these versions. This base metric will still work as it did
        prior to v0.10 until v0.11. From v0.11 the `task` argument introduced in this metric will be required
        and the general order of arguments may change, such that this metric will just function as an single
        entrypoint to calling the three specialized versions.

    Calculates `Matthews correlation coefficient`_ that measures
    the general correlation or quality of a classification. In the binary case it
    is defined as:

    .. math::
        MCC = \frac{TP*TN - FP*FN}{\sqrt{(TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)}}

    where TP, TN, FP and FN are respectively the true postitives, true negatives,
    false positives and false negatives. Also works in the case of multi-label or
    multi-class input.

    Args:
        preds: (float or long tensor), Either a ``(N, ...)`` tensor with labels or
            ``(N, C, ...)`` where C is the number of classes, tensor with labels/probabilities
        target: ``target`` (long tensor), tensor with shape ``(N, ...)`` with ground true labels
        num_classes: Number of classes in the dataset.
        threshold:
            Threshold value for binary or multi-label probabilities.

    Example:
        >>> from torchmetrics.functional import matthews_corrcoef
        >>> target = torch.tensor([1, 1, 0, 0])
        >>> preds = torch.tensor([0, 1, 0, 0])
        >>> matthews_corrcoef(preds, target, num_classes=2)
        tensor(0.5774)
    """
    if task is not None:
        if task == "binary":
            return binary_matthews_corrcoef(preds, target, threshold, ignore_index, validate_args)
        if task == "multiclass":
            assert isinstance(num_classes, int)
            return multiclass_matthews_corrcoef(preds, target, num_classes, ignore_index, validate_args)
        if task == "multilabel":
            assert isinstance(num_labels, int)
            return multilabel_matthews_corrcoef(preds, target, num_labels, threshold, ignore_index, validate_args)
        raise ValueError(
            f"Expected argument `task` to either be `'binary'`, `'multiclass'` or `'multilabel'` but got {task}"
        )
    else:
        rank_zero_warn(
            "From v0.10 an `'binary_*'`, `'multiclass_*'`, `'multilabel_*'` version now exist of each classification"
            " metric. Moving forward we recommend using these versions. This base metric will still work as it did"
            " prior to v0.10 until v0.11. From v0.11 the `task` argument introduced in this metric will be required"
            " and the general order of arguments may change, such that this metric will just function as an single"
            " entrypoint to calling the three specialized versions.",
            DeprecationWarning,
        )
    confmat = _matthews_corrcoef_update(preds, target, num_classes, threshold)
    return _matthews_corrcoef_compute(confmat)
