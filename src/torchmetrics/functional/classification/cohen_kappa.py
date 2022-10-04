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
    _confusion_matrix_compute,
    _confusion_matrix_update,
    _multiclass_confusion_matrix_arg_validation,
    _multiclass_confusion_matrix_format,
    _multiclass_confusion_matrix_tensor_validation,
    _multiclass_confusion_matrix_update,
)
from torchmetrics.utilities.prints import rank_zero_warn


def _cohen_kappa_reduce(confmat: Tensor, weights: Optional[Literal["linear", "quadratic", "none"]] = None) -> Tensor:
    """Reduce an un-normalized confusion matrix of shape (n_classes, n_classes) into the cohen kappa score."""
    confmat = confmat.float() if not confmat.is_floating_point() else confmat
    n_classes = confmat.shape[0]
    sum0 = confmat.sum(dim=0, keepdim=True)
    sum1 = confmat.sum(dim=1, keepdim=True)
    expected = sum1 @ sum0 / sum0.sum()  # outer product

    if weights is None or weights == "none":
        w_mat = torch.ones_like(confmat).flatten()
        w_mat[:: n_classes + 1] = 0
        w_mat = w_mat.reshape(n_classes, n_classes)
    elif weights in ("linear", "quadratic"):
        w_mat = torch.zeros_like(confmat)
        w_mat += torch.arange(n_classes, dtype=w_mat.dtype, device=w_mat.device)
        if weights == "linear":
            w_mat = torch.abs(w_mat - w_mat.T)
        else:
            w_mat = torch.pow(w_mat - w_mat.T, 2.0)
    else:
        raise ValueError(
            f"Received {weights} for argument ``weights`` but should be either" " None, 'linear' or 'quadratic'"
        )
    k = torch.sum(w_mat * confmat) / torch.sum(w_mat * expected)
    return 1 - k


def _binary_cohen_kappa_arg_validation(
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
    weights: Optional[Literal["linear", "quadratic", "none"]] = None,
) -> None:
    """Validate non tensor input.

    - ``threshold`` has to be a float in the [0,1] range
    - ``ignore_index`` has to be None or int
    - ``weights`` has to be "linear" | "quadratic" | "none" | None
    """
    _binary_confusion_matrix_arg_validation(threshold, ignore_index, normalize=None)
    allowed_weights = ("linear", "quadratic", "none", None)
    if weights not in allowed_weights:
        raise ValueError(f"Expected argument `weight` to be one of {allowed_weights}, but got {weights}.")


def binary_cohen_kappa(
    preds: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    weights: Optional[Literal["linear", "quadratic", "none"]] = None,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    r"""Calculates `Cohen's kappa score`_ that measures inter-annotator agreement for binary tasks. It is defined
    as.

    .. math::
        \kappa = (p_o - p_e) / (1 - p_e)

    where :math:`p_o` is the empirical probability of agreement and :math:`p_e` is
    the expected agreement when both annotators assign labels randomly. Note that
    :math:`p_e` is estimated using a per-annotator empirical prior over the
    class labels.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Addtionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, ...)``

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        threshold: Threshold for transforming probability to binary (0,1) predictions
        weights: Weighting type to calculate the score. Choose from:

            - ``None`` or ``'none'``: no weighting
            - ``'linear'``: linear weighting
            - ``'quadratic'``: quadratic weighting
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (preds is int tensor):
        >>> from torchmetrics.functional.classification import binary_cohen_kappa
        >>> target = torch.tensor([1, 1, 0, 0])
        >>> preds = torch.tensor([0, 1, 0, 0])
        >>> binary_cohen_kappa(preds, target)
        tensor(0.5000)

    Example (preds is float tensor):
        >>> from torchmetrics.functional.classification import binary_cohen_kappa
        >>> target = torch.tensor([1, 1, 0, 0])
        >>> preds = torch.tensor([0.35, 0.85, 0.48, 0.01])
        >>> binary_cohen_kappa(preds, target)
        tensor(0.5000)
    """
    if validate_args:
        _binary_cohen_kappa_arg_validation(threshold, ignore_index, weights)
        _binary_confusion_matrix_tensor_validation(preds, target, ignore_index)
    preds, target = _binary_confusion_matrix_format(preds, target, threshold, ignore_index)
    confmat = _binary_confusion_matrix_update(preds, target)
    return _cohen_kappa_reduce(confmat, weights)


def _multiclass_cohen_kappa_arg_validation(
    num_classes: int,
    ignore_index: Optional[int] = None,
    weights: Optional[Literal["linear", "quadratic", "none"]] = None,
) -> None:
    """Validate non tensor input.

    - ``num_classes`` has to be a int larger than 1
    - ``ignore_index`` has to be None or int
    - ``weights`` has to be "linear" | "quadratic" | "none" | None
    """
    _multiclass_confusion_matrix_arg_validation(num_classes, ignore_index, normalize=None)
    allowed_weights = ("linear", "quadratic", "none", None)
    if weights not in allowed_weights:
        raise ValueError(f"Expected argument `weight` to be one of {allowed_weights}, but got {weights}.")


def multiclass_cohen_kappa(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    weights: Optional[Literal["linear", "quadratic", "none"]] = None,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    r"""Calculates `Cohen's kappa score`_ that measures inter-annotator agreement for multiclass tasks. It is
    defined as.

    .. math::
        \kappa = (p_o - p_e) / (1 - p_e)

    where :math:`p_o` is the empirical probability of agreement and :math:`p_e` is
    the expected agreement when both annotators assign labels randomly. Note that
    :math:`p_e` is estimated using a per-annotator empirical prior over the
    class labels.

    Accepts the following input tensors:

    - ``preds``: ``(N, ...)`` (int tensor) or ``(N, C, ..)`` (float tensor). If preds is a floating point
      we apply ``torch.argmax`` along the ``C`` dimension to automatically convert probabilities/logits into
      an int tensor.
    - ``target`` (int tensor): ``(N, ...)``

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_classes: Integer specifing the number of classes
        weights: Weighting type to calculate the score. Choose from:

            - ``None`` or ``'none'``: no weighting
            - ``'linear'``: linear weighting
            - ``'quadratic'``: quadratic weighting


        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (pred is integer tensor):
        >>> from torchmetrics.functional.classification import multiclass_cohen_kappa
        >>> target = torch.tensor([2, 1, 0, 0])
        >>> preds = torch.tensor([2, 1, 0, 1])
        >>> multiclass_cohen_kappa(preds, target, num_classes=3)
        tensor(0.6364)

    Example (pred is float tensor):
        >>> from torchmetrics.functional.classification import multiclass_cohen_kappa
        >>> target = torch.tensor([2, 1, 0, 0])
        >>> preds = torch.tensor([
        ...   [0.16, 0.26, 0.58],
        ...   [0.22, 0.61, 0.17],
        ...   [0.71, 0.09, 0.20],
        ...   [0.05, 0.82, 0.13],
        ... ])
        >>> multiclass_cohen_kappa(preds, target, num_classes=3)
        tensor(0.6364)
    """
    if validate_args:
        _multiclass_cohen_kappa_arg_validation(num_classes, ignore_index, weights)
        _multiclass_confusion_matrix_tensor_validation(preds, target, num_classes, ignore_index)
    preds, target = _multiclass_confusion_matrix_format(preds, target, ignore_index)
    confmat = _multiclass_confusion_matrix_update(preds, target, num_classes)
    return _cohen_kappa_reduce(confmat, weights)


_cohen_kappa_update = _confusion_matrix_update


def _cohen_kappa_compute(confmat: Tensor, weights: Optional[str] = None) -> Tensor:
    """Computes Cohen's kappa based on the weighting type.

    Args:
        confmat: Confusion matrix without normalization
        weights: Weighting type to calculate the score. Choose from:

            - ``None`` or ``'none'``: no weighting
            - ``'linear'``: linear weighting
            - ``'quadratic'``: quadratic weighting

    Example:
        >>> target = torch.tensor([1, 1, 0, 0])
        >>> preds = torch.tensor([0, 1, 0, 0])
        >>> confmat = _cohen_kappa_update(preds, target, num_classes=2)
        >>> _cohen_kappa_compute(confmat)
        tensor(0.5000)
    """

    confmat = _confusion_matrix_compute(confmat)
    confmat = confmat.float() if not confmat.is_floating_point() else confmat
    n_classes = confmat.shape[0]
    sum0 = confmat.sum(dim=0, keepdim=True)
    sum1 = confmat.sum(dim=1, keepdim=True)
    expected = sum1 @ sum0 / sum0.sum()  # outer product

    if weights is None:
        w_mat = torch.ones_like(confmat).flatten()
        w_mat[:: n_classes + 1] = 0
        w_mat = w_mat.reshape(n_classes, n_classes)
    elif weights in ("linear", "quadratic"):
        w_mat = torch.zeros_like(confmat)
        w_mat += torch.arange(n_classes, dtype=w_mat.dtype, device=w_mat.device)
        if weights == "linear":
            w_mat = torch.abs(w_mat - w_mat.T)
        else:
            w_mat = torch.pow(w_mat - w_mat.T, 2.0)
    else:
        raise ValueError(
            f"Received {weights} for argument ``weights`` but should be either" " None, 'linear' or 'quadratic'"
        )

    k = torch.sum(w_mat * confmat) / torch.sum(w_mat * expected)
    return 1 - k


def cohen_kappa(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    weights: Optional[Literal["linear", "quadratic", "none"]] = None,
    threshold: float = 0.5,
    task: Optional[Literal["binary", "multiclass", "multilabel"]] = None,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    r"""Cohen's kappa.

    .. note::
        From v0.10 an ``'binary_*'``, ``'multiclass_*'``, ``'multilabel_*'`` version now exist of each classification
        metric. Moving forward we recommend using these versions. This base metric will still work as it did
        prior to v0.10 until v0.11. From v0.11 the `task` argument introduced in this metric will be required
        and the general order of arguments may change, such that this metric will just function as an single
        entrypoint to calling the three specialized versions.


    Calculates `Cohen's kappa score`_ that measures inter-annotator agreement.

    It is defined as

    .. math::
        \kappa = (p_o - p_e) / (1 - p_e)

    where :math:`p_o` is the empirical probability of agreement and :math:`p_e` is
    the expected agreement when both annotators assign labels randomly. Note that
    :math:`p_e` is estimated using a per-annotator empirical prior over the
    class labels.

    Args:
        preds: (float or long tensor), Either a ``(N, ...)`` tensor with labels or
            ``(N, C, ...)`` where C is the number of classes, tensor with labels/probabilities
        target: ``target`` (long tensor), tensor with shape ``(N, ...)`` with ground true labels
        num_classes: Number of classes in the dataset.
        weights: Weighting type to calculate the score. Choose from:

            - ``None`` or ``'none'``: no weighting
            - ``'linear'``: linear weighting
            - ``'quadratic'``: quadratic weighting

        threshold: Threshold value for binary or multi-label probabilities.

    Example:
        >>> from torchmetrics.functional import cohen_kappa
        >>> target = torch.tensor([1, 1, 0, 0])
        >>> preds = torch.tensor([0, 1, 0, 0])
        >>> cohen_kappa(preds, target, num_classes=2)
        tensor(0.5000)
    """
    if task is not None:
        if task == "binary":
            return binary_cohen_kappa(preds, target, threshold, weights, ignore_index, validate_args)
        if task == "multiclass":
            return multiclass_cohen_kappa(preds, target, num_classes, weights, ignore_index, validate_args)
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

    confmat = _cohen_kappa_update(preds, target, num_classes, threshold)
    return _cohen_kappa_compute(confmat, weights)
