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
    ignore_index: Optional[int] = None,
    weights: Optional[Literal["linear", "quadratic", "none"]] = None,
    validate_args: bool = True,
) -> Tensor:
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
    ignore_index: Optional[int] = None,
    weights: Optional[Literal["linear", "quadratic", "none"]] = None,
    validate_args: bool = True,
) -> Tensor:
    if validate_args:
        _multiclass_cohen_kappa_arg_validation(num_classes, ignore_index, weights)
        _multiclass_confusion_matrix_tensor_validation(preds, target, num_classes, ignore_index)
    preds, target = _multiclass_confusion_matrix_format(preds, target, ignore_index)
    confmat = _multiclass_confusion_matrix_update(preds, target, num_classes)
    return _cohen_kappa_reduce(confmat, weights)


# -------------------------- Old stuff --------------------------

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
    weights: Optional[str] = None,
    threshold: float = 0.5,
) -> Tensor:
    r"""Calculates `Cohen's kappa score`_ that measures inter-annotator agreement.

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
    confmat = _cohen_kappa_update(preds, target, num_classes, threshold)
    return _cohen_kappa_compute(confmat, weights)
