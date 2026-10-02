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
from torchmetrics.utilities.enums import ClassificationTask


def _matthews_corrcoef_arg_validation(zero_division: Optional[float] = None) -> None:
    """Validate non tensor input specific to the matthews corrcoef metric.

    - ``zero_division`` has to be ``None`` or a float in the [-1, 1] range

    """
    if zero_division is None:
        return
    if isinstance(zero_division, bool) or not isinstance(zero_division, (int, float)) or not -1 <= zero_division <= 1:
        raise ValueError(
            f"Expected argument `zero_division` to be `None` or a float in the [-1, 1] range, but got {zero_division}."
        )


def _matthews_corrcoef_reduce(confmat: Tensor, zero_division: Optional[float] = None) -> Tensor:
    """Reduce an un-normalized confusion matrix of shape (n_classes, n_classes) into the matthews corrcoef score.

    See: https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-019-6413-7 for more info.

    Args:
        confmat: un-normalized confusion matrix
        zero_division: value returned when the score is undefined, i.e. when all targets or all predictions belong to a
            single class. If ``None``, the default convention is used, see the public functions for details.

    """
    # convert multilabel into binary
    confmat = confmat.sum(0) if confmat.ndim == 3 else confmat

    # The score is undefined when all targets (rows) or all predictions (columns) belong to a single class. This is
    # checked on the counts rather than on the computed denominator, which can underflow to zero in low precision
    # dtypes for confusion matrices that are not degenerate.
    if zero_division is not None and ((confmat.sum(dim=-1) != 0).sum() <= 1 or (confmat.sum(dim=-2) != 0).sum() <= 1):
        return torch.tensor(zero_division, dtype=torch.float32, device=confmat.device)

    if confmat.numel() == 4:  # binary case
        tn, fp, fn, tp = confmat.reshape(-1)
        if tp + tn != 0 and fp + fn == 0:
            return torch.tensor(1.0, dtype=confmat.dtype, device=confmat.device)

        if tp + tn == 0 and fp + fn != 0:
            return torch.tensor(-1.0, dtype=confmat.dtype, device=confmat.device)

    confmat = confmat.float()
    tk = confmat.sum(dim=-1)
    pk = confmat.sum(dim=-2)
    c = torch.trace(confmat)
    s = confmat.sum()

    cov_ytyp = c * s - sum(tk * pk)
    cov_ypyp = s**2 - sum(pk * pk)
    cov_ytyt = s**2 - sum(tk * tk)

    numerator = cov_ytyp
    denom = cov_ypyp * cov_ytyt

    if denom == 0 and confmat.numel() == 4:
        eps = torch.tensor(torch.finfo(torch.float32).eps, dtype=torch.float32, device=confmat.device)
        if fn == 0 and tn == 0:
            numerator = torch.sqrt(eps) * (tp - fp)
        elif fp == 0 and tn == 0:
            numerator = torch.sqrt(eps) * (tp - fn)
        elif tp == 0 and fn == 0:
            numerator = torch.sqrt(eps) * (tn - fp)
        elif tp == 0 and fp == 0:
            numerator = torch.sqrt(eps) * (tn - fn)
        elif tp == 0:
            numerator = tn - fp * fn
        elif tn == 0:
            numerator = tp - fp * fn
        elif fp == 0 or fn == 0:
            numerator = tp * tn
        else:
            return torch.tensor(0, dtype=confmat.dtype, device=confmat.device)
        denom = (tp + fp + eps) * (tp + fn + eps) * (tn + fp + eps) * (tn + fn + eps)
    elif denom == 0:
        return torch.tensor(0, dtype=confmat.dtype, device=confmat.device)
    return numerator / torch.sqrt(denom)


def binary_matthews_corrcoef(
    preds: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
    zero_division: Optional[float] = None,
) -> Tensor:
    r"""Calculate `Matthews correlation coefficient`_ for binary tasks.

    This metric measures the general correlation or quality of a classification.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Additionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, ...)``

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        threshold: Threshold for transforming probability to binary (0,1) predictions
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        zero_division:
            Value returned when the score is undefined, i.e. when all targets or all predictions belong to a single
            class. Should be ``None`` or a float in the [-1, 1] range. If ``None`` (default), the existing behaviour
            is kept: for binary and multilabel input the score is ``1`` when all samples are predicted correctly,
            ``-1`` when all are predicted wrongly and the remaining undefined cases are approximated with a small
            epsilon; for multiclass input undefined cases give ``0``. Set to ``0`` to match
            ``sklearn.metrics.matthews_corrcoef``.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (preds is int tensor):
        >>> from torch import tensor
        >>> from torchmetrics.functional.classification import binary_matthews_corrcoef
        >>> target = tensor([1, 1, 0, 0])
        >>> preds = tensor([0, 1, 0, 0])
        >>> binary_matthews_corrcoef(preds, target)
        tensor(0.5774)

    Example (preds is float tensor):
        >>> from torchmetrics.functional.classification import binary_matthews_corrcoef
        >>> target = tensor([1, 1, 0, 0])
        >>> preds = tensor([0.35, 0.85, 0.48, 0.01])
        >>> binary_matthews_corrcoef(preds, target)
        tensor(0.5774)

    """
    if validate_args:
        _matthews_corrcoef_arg_validation(zero_division)
        _binary_confusion_matrix_arg_validation(threshold, ignore_index, normalize=None)
        _binary_confusion_matrix_tensor_validation(preds, target, ignore_index)
    preds, target = _binary_confusion_matrix_format(preds, target, threshold, ignore_index)
    confmat = _binary_confusion_matrix_update(preds, target)
    return _matthews_corrcoef_reduce(confmat, zero_division)


def multiclass_matthews_corrcoef(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
    zero_division: Optional[float] = None,
) -> Tensor:
    r"""Calculate `Matthews correlation coefficient`_ for multiclass tasks.

    This metric measures the general correlation or quality of a classification.

    Accepts the following input tensors:

    - ``preds``: ``(N, ...)`` (int tensor) or ``(N, C, ..)`` (float tensor). If preds is a floating point
      we apply ``torch.argmax`` along the ``C`` dimension to automatically convert probabilities/logits into
      an int tensor.
    - ``target`` (int tensor): ``(N, ...)``

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_classes: Integer specifying the number of classes
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        zero_division:
            Value returned when the score is undefined, i.e. when all targets or all predictions belong to a single
            class. Should be ``None`` or a float in the [-1, 1] range. If ``None`` (default), the existing behaviour
            is kept: for binary and multilabel input the score is ``1`` when all samples are predicted correctly,
            ``-1`` when all are predicted wrongly and the remaining undefined cases are approximated with a small
            epsilon; for multiclass input undefined cases give ``0``. Set to ``0`` to match
            ``sklearn.metrics.matthews_corrcoef``.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (pred is integer tensor):
        >>> from torch import tensor
        >>> from torchmetrics.functional.classification import multiclass_matthews_corrcoef
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([2, 1, 0, 1])
        >>> multiclass_matthews_corrcoef(preds, target, num_classes=3)
        tensor(0.7000)

    Example (pred is float tensor):
        >>> from torchmetrics.functional.classification import multiclass_matthews_corrcoef
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([[0.16, 0.26, 0.58],
        ...                 [0.22, 0.61, 0.17],
        ...                 [0.71, 0.09, 0.20],
        ...                 [0.05, 0.82, 0.13]])
        >>> multiclass_matthews_corrcoef(preds, target, num_classes=3)
        tensor(0.7000)

    """
    if validate_args:
        _matthews_corrcoef_arg_validation(zero_division)
        _multiclass_confusion_matrix_arg_validation(num_classes, ignore_index, normalize=None)
        _multiclass_confusion_matrix_tensor_validation(preds, target, num_classes, ignore_index)
    preds, target = _multiclass_confusion_matrix_format(preds, target, ignore_index)
    confmat = _multiclass_confusion_matrix_update(preds, target, num_classes)
    return _matthews_corrcoef_reduce(confmat, zero_division)


def multilabel_matthews_corrcoef(
    preds: Tensor,
    target: Tensor,
    num_labels: int,
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
    zero_division: Optional[float] = None,
) -> Tensor:
    r"""Calculate `Matthews correlation coefficient`_ for multilabel tasks.

    This metric measures the general correlation or quality of a classification.

    Accepts the following input tensors:

        - ``preds`` (int or float tensor): ``(N, C, ...)``. If preds is a floating point tensor with values outside
          [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Additionally,
          we convert to int tensor with thresholding using the value in ``threshold``.
        - ``target`` (int tensor): ``(N, C, ...)``

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_labels: Integer specifying the number of labels
        threshold: Threshold for transforming probability to binary (0,1) predictions
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        zero_division:
            Value returned when the score is undefined, i.e. when all targets or all predictions belong to a single
            class. Should be ``None`` or a float in the [-1, 1] range. If ``None`` (default), the existing behaviour
            is kept: for binary and multilabel input the score is ``1`` when all samples are predicted correctly,
            ``-1`` when all are predicted wrongly and the remaining undefined cases are approximated with a small
            epsilon; for multiclass input undefined cases give ``0``. Set to ``0`` to match
            ``sklearn.metrics.matthews_corrcoef``.

    Example (preds is int tensor):
        >>> from torch import tensor
        >>> from torchmetrics.functional.classification import multilabel_matthews_corrcoef
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0, 0, 1], [1, 0, 1]])
        >>> multilabel_matthews_corrcoef(preds, target, num_labels=3)
        tensor(0.3333)

    Example (preds is float tensor):
        >>> from torchmetrics.functional.classification import multilabel_matthews_corrcoef
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0.11, 0.22, 0.84], [0.73, 0.33, 0.92]])
        >>> multilabel_matthews_corrcoef(preds, target, num_labels=3)
        tensor(0.3333)

    """
    if validate_args:
        _matthews_corrcoef_arg_validation(zero_division)
        _multilabel_confusion_matrix_arg_validation(num_labels, threshold, ignore_index, normalize=None)
        _multilabel_confusion_matrix_tensor_validation(preds, target, num_labels, ignore_index)
    preds, target = _multilabel_confusion_matrix_format(preds, target, num_labels, threshold, ignore_index)
    confmat = _multilabel_confusion_matrix_update(preds, target, num_labels)
    return _matthews_corrcoef_reduce(confmat, zero_division)


def matthews_corrcoef(
    preds: Tensor,
    target: Tensor,
    task: Literal["binary", "multiclass", "multilabel"],
    threshold: float = 0.5,
    num_classes: Optional[int] = None,
    num_labels: Optional[int] = None,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
    zero_division: Optional[float] = None,
) -> Tensor:
    r"""Calculate `Matthews correlation coefficient`_ .

    This metric measures the general correlation or quality of a classification.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``'multilabel'``. See the documentation of
    :func:`~torchmetrics.functional.classification.binary_matthews_corrcoef`,
    :func:`~torchmetrics.functional.classification.multiclass_matthews_corrcoef` and
    :func:`~torchmetrics.functional.classification.multilabel_matthews_corrcoef` for
    the specific details of each argument influence and examples.

    Legacy Example:
        >>> from torch import tensor
        >>> target = tensor([1, 1, 0, 0])
        >>> preds = tensor([0, 1, 0, 0])
        >>> matthews_corrcoef(preds, target, task="multiclass", num_classes=2)
        tensor(0.5774)

    """
    task = ClassificationTask.from_str(task)
    if task == ClassificationTask.BINARY:
        return binary_matthews_corrcoef(preds, target, threshold, ignore_index, validate_args, zero_division)
    if task == ClassificationTask.MULTICLASS:
        if not isinstance(num_classes, int):
            raise ValueError(f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`")
        return multiclass_matthews_corrcoef(preds, target, num_classes, ignore_index, validate_args, zero_division)
    if task == ClassificationTask.MULTILABEL:
        if not isinstance(num_labels, int):
            raise ValueError(f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`")
        return multilabel_matthews_corrcoef(
            preds, target, num_labels, threshold, ignore_index, validate_args, zero_division
        )
    raise ValueError(f"Not handled value: {task}")
