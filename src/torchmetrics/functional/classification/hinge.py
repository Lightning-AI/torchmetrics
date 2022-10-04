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
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, tensor
from typing_extensions import Literal

from torchmetrics.functional.classification.confusion_matrix import (
    _binary_confusion_matrix_format,
    _binary_confusion_matrix_tensor_validation,
    _multiclass_confusion_matrix_format,
    _multiclass_confusion_matrix_tensor_validation,
)
from torchmetrics.utilities.checks import _input_squeeze
from torchmetrics.utilities.data import to_onehot
from torchmetrics.utilities.enums import DataType, EnumStr
from torchmetrics.utilities.prints import rank_zero_warn


def _hinge_loss_compute(measure: Tensor, total: Tensor) -> Tensor:
    return measure / total


def _binary_hinge_loss_arg_validation(squared: bool, ignore_index: Optional[int] = None) -> None:
    if not isinstance(squared, bool):
        raise ValueError(f"Expected argument `squared` to be an bool but got {squared}")
    if ignore_index is not None and not isinstance(ignore_index, int):
        raise ValueError(f"Expected argument `ignore_index` to either be `None` or an integer, but got {ignore_index}")


def _binary_hinge_loss_tensor_validation(preds: Tensor, target: Tensor, ignore_index: Optional[int] = None) -> None:
    _binary_confusion_matrix_tensor_validation(preds, target, ignore_index)
    if not preds.is_floating_point():
        raise ValueError(
            "Expected argument `preds` to be floating tensor with probabilities/logits"
            f" but got tensor with dtype {preds.dtype}"
        )


def _binary_hinge_loss_update(
    preds: Tensor,
    target: Tensor,
    squared: bool,
) -> Tuple[Tensor, Tensor]:

    target = target.bool()
    margin = torch.zeros_like(preds)
    margin[target] = preds[target]
    margin[~target] = -preds[~target]

    measures = 1 - margin
    measures = torch.clamp(measures, 0)

    if squared:
        measures = measures.pow(2)

    total = tensor(target.shape[0], device=target.device)
    return measures.sum(dim=0), total


def binary_hinge_loss(
    preds: Tensor,
    target: Tensor,
    squared: bool = False,
    ignore_index: Optional[int] = None,
    validate_args: bool = False,
) -> Tensor:
    r"""Computes the mean `Hinge loss`_ typically used for Support Vector Machines (SVMs) for binary tasks. It is
    defined as:

    .. math::
        \text{Hinge loss} = \max(0, 1 - y \times \hat{y})

    Where :math:`y \in {-1, 1}` is the target, and :math:`\hat{y} \in \mathbb{R}` is the prediction.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (int tensor): ``(N, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain {0,1} values (except if `ignore_index` is specified).

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        squared:
            If True, this will compute the squared hinge loss. Otherwise, computes the regular hinge loss.
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Example:
        >>> from torchmetrics.functional.classification import binary_hinge_loss
        >>> preds = torch.tensor([0.25, 0.25, 0.55, 0.75, 0.75])
        >>> target = torch.tensor([0, 0, 1, 1, 1])
        >>> binary_hinge_loss(preds, target)
        tensor(0.6900)
        >>> binary_hinge_loss(preds, target, squared=True)
        tensor(0.6905)
    """
    if validate_args:
        _binary_hinge_loss_arg_validation(squared, ignore_index)
        _binary_hinge_loss_tensor_validation(preds, target, ignore_index)
    preds, target = _binary_confusion_matrix_format(
        preds, target, threshold=0.0, ignore_index=ignore_index, convert_to_labels=False
    )
    measures, total = _binary_hinge_loss_update(preds, target, squared)
    return _hinge_loss_compute(measures, total)


def _multiclass_hinge_loss_arg_validation(
    num_classes: int,
    squared: bool = False,
    multiclass_mode: Literal["crammer-singer", "one-vs-all"] = "crammer-singer",
    ignore_index: Optional[int] = None,
) -> None:
    _binary_hinge_loss_arg_validation(squared, ignore_index)
    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError(f"Expected argument `num_classes` to be an integer larger than 1, but got {num_classes}")
    allowed_mm = ("crammer-singer", "one-vs-all")
    if multiclass_mode not in allowed_mm:
        raise ValueError(f"Expected argument `multiclass_mode` to be one of {allowed_mm}, but got {multiclass_mode}.")


def _multiclass_hinge_loss_tensor_validation(
    preds: Tensor, target: Tensor, num_classes: int, ignore_index: Optional[int] = None
) -> None:
    _multiclass_confusion_matrix_tensor_validation(preds, target, num_classes, ignore_index)
    if not preds.is_floating_point():
        raise ValueError(
            "Expected argument `preds` to be floating tensor with probabilities/logits"
            f" but got tensor with dtype {preds.dtype}"
        )


def _multiclass_hinge_loss_update(
    preds: Tensor,
    target: Tensor,
    squared: bool,
    multiclass_mode: Literal["crammer-singer", "one-vs-all"] = "crammer-singer",
) -> Tuple[Tensor, Tensor]:
    if not torch.all((0 <= preds) * (preds <= 1)):
        preds = preds.softmax(1)

    target = to_onehot(target, max(2, preds.shape[1])).bool()
    if multiclass_mode == "crammer-singer":
        margin = preds[target]
        margin -= torch.max(preds[~target].view(preds.shape[0], -1), dim=1)[0]
    else:
        target = target.bool()
        margin = torch.zeros_like(preds)
        margin[target] = preds[target]
        margin[~target] = -preds[~target]

    measures = 1 - margin
    measures = torch.clamp(measures, 0)

    if squared:
        measures = measures.pow(2)

    total = tensor(target.shape[0], device=target.device)
    return measures.sum(dim=0), total


def multiclass_hinge_loss(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    squared: bool = False,
    multiclass_mode: Literal["crammer-singer", "one-vs-all"] = "crammer-singer",
    ignore_index: Optional[int] = None,
    validate_args: bool = False,
) -> Tensor:
    r"""Computes the mean `Hinge loss`_ typically used for Support Vector Machines (SVMs) for multiclass tasks.

    The metric can be computed in two ways. Either, the definition by Crammer and Singer is used:

    .. math::
        \text{Hinge loss} = \max\left(0, 1 - \hat{y}_y + \max_{i \ne y} (\hat{y}_i)\right)

    Where :math:`y \in {0, ..., \mathrm{C}}` is the target class (where :math:`\mathrm{C}` is the number of classes),
    and :math:`\hat{y} \in \mathbb{R}^\mathrm{C}` is the predicted output per class. Alternatively, the metric can
    also be computed in one-vs-all approach, where each class is valued against all other classes in a binary fashion.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, C, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      softmax per sample.
    - ``target`` (int tensor): ``(N, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain values in the [0, n_classes-1] range (except if `ignore_index` is specified).

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_classes: Integer specifing the number of classes
        squared:
            If True, this will compute the squared hinge loss. Otherwise, computes the regular hinge loss.
        multiclass_mode:
            Determines how to compute the metric
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Example:
        >>> from torchmetrics.functional.classification import multiclass_hinge_loss
        >>> preds = torch.tensor([[0.25, 0.20, 0.55],
        ...                       [0.55, 0.05, 0.40],
        ...                       [0.10, 0.30, 0.60],
        ...                       [0.90, 0.05, 0.05]])
        >>> target = torch.tensor([0, 1, 2, 0])
        >>> multiclass_hinge_loss(preds, target, num_classes=3)
        tensor(0.9125)
        >>> multiclass_hinge_loss(preds, target, num_classes=3, squared=True)
        tensor(1.1131)
        >>> multiclass_hinge_loss(preds, target, num_classes=3, multiclass_mode='one-vs-all')
        tensor([0.8750, 1.1250, 1.1000])
    """
    if validate_args:
        _multiclass_hinge_loss_arg_validation(num_classes, squared, multiclass_mode, ignore_index)
        _multiclass_hinge_loss_tensor_validation(preds, target, num_classes, ignore_index)
    preds, target = _multiclass_confusion_matrix_format(preds, target, ignore_index, convert_to_labels=False)
    measures, total = _multiclass_hinge_loss_update(preds, target, squared, multiclass_mode)
    return _hinge_loss_compute(measures, total)


class MulticlassMode(EnumStr):
    """Enum to represent possible multiclass modes of hinge.

    >>> "Crammer-Singer" in list(MulticlassMode)
    True
    """

    CRAMMER_SINGER = "crammer-singer"
    ONE_VS_ALL = "one-vs-all"


def _check_shape_and_type_consistency_hinge(
    preds: Tensor,
    target: Tensor,
) -> DataType:
    """Checks shape and type of ``preds`` and ``target`` and returns mode of the input tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor

    Raises:
        `ValueError`: if ``target`` is not one dimensional
        `ValueError`: if ``preds`` and ``target`` do not have the same shape in the first dimension
        `ValueError`: if ``preds`` is neither one nor two-dimensional
    """

    if target.ndim > 1:
        raise ValueError(
            f"The `target` should be one dimensional, got `target` with shape={target.shape}.",
        )

    if preds.ndim == 1:
        if preds.shape != target.shape:
            raise ValueError(
                "The `preds` and `target` should have the same shape,",
                f" got `preds` with shape={preds.shape} and `target` with shape={target.shape}.",
            )
        mode = DataType.BINARY
    elif preds.ndim == 2:
        if preds.shape[0] != target.shape[0]:
            raise ValueError(
                "The `preds` and `target` should have the same shape in the first dimension,",
                f" got `preds` with shape={preds.shape} and `target` with shape={target.shape}.",
            )
        mode = DataType.MULTICLASS
    else:
        raise ValueError(f"The `preds` should be one or two dimensional, got `preds` with shape={preds.shape}.")
    return mode


def _hinge_update(
    preds: Tensor,
    target: Tensor,
    squared: bool = False,
    multiclass_mode: Optional[Union[str, MulticlassMode]] = None,
) -> Tuple[Tensor, Tensor]:
    """Updates and returns sum over Hinge loss scores for each observation and the total number of observations.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        squared: If True, this will compute the squared hinge loss. Otherwise, computes the regular hinge loss.
        multiclass_mode:
            Which approach to use for multi-class inputs (has no effect in the binary case). ``None`` (default),
            ``MulticlassMode.CRAMMER_SINGER`` or ``"crammer-singer"``, uses the Crammer Singer multi-class hinge loss.
            ``MulticlassMode.ONE_VS_ALL`` or ``"one-vs-all"`` computes the hinge loss in a one-vs-all fashion.
    """
    preds, target = _input_squeeze(preds, target)

    mode = _check_shape_and_type_consistency_hinge(preds, target)

    if mode == DataType.MULTICLASS:
        target = to_onehot(target, max(2, preds.shape[1])).bool()

    if mode == DataType.MULTICLASS and (multiclass_mode is None or multiclass_mode == MulticlassMode.CRAMMER_SINGER):
        margin = preds[target]
        margin -= torch.max(preds[~target].view(preds.shape[0], -1), dim=1)[0]
    elif mode == DataType.BINARY or multiclass_mode == MulticlassMode.ONE_VS_ALL:
        target = target.bool()
        margin = torch.zeros_like(preds)
        margin[target] = preds[target]
        margin[~target] = -preds[~target]
    else:
        raise ValueError(
            "The `multiclass_mode` should be either None / 'crammer-singer' / MulticlassMode.CRAMMER_SINGER"
            "(default) or 'one-vs-all' / MulticlassMode.ONE_VS_ALL,"
            f" got {multiclass_mode}."
        )

    measures = 1 - margin
    measures = torch.clamp(measures, 0)

    if squared:
        measures = measures.pow(2)

    total = tensor(target.shape[0], device=target.device)
    return measures.sum(dim=0), total


def _hinge_compute(measure: Tensor, total: Tensor) -> Tensor:
    """Computes mean Hinge loss.

    Args:
        measure: Sum over hinge losses for each observation
        total: Number of observations

    Example:
        >>> # binary case
        >>> target = torch.tensor([0, 1, 1])
        >>> preds = torch.tensor([-2.2, 2.4, 0.1])
        >>> measure, total = _hinge_update(preds, target)
        >>> _hinge_compute(measure, total)
        tensor(0.3000)

        >>> # multiclass case
        >>> target = torch.tensor([0, 1, 2])
        >>> preds = torch.tensor([[-1.0, 0.9, 0.2], [0.5, -1.1, 0.8], [2.2, -0.5, 0.3]])
        >>> measure, total = _hinge_update(preds, target)
        >>> _hinge_compute(measure, total)
        tensor(2.9000)

        >>> # multiclass one-vs-all mode case
        >>> target = torch.tensor([0, 1, 2])
        >>> preds = torch.tensor([[-1.0, 0.9, 0.2], [0.5, -1.1, 0.8], [2.2, -0.5, 0.3]])
        >>> measure, total = _hinge_update(preds, target, multiclass_mode="one-vs-all")
        >>> _hinge_compute(measure, total)
        tensor([2.2333, 1.5000, 1.2333])
    """

    return measure / total


def hinge_loss(
    preds: Tensor,
    target: Tensor,
    squared: bool = False,
    multiclass_mode: Optional[Literal["crammer-singer", "one-vs-all"]] = None,
    task: Optional[Literal["binary", "multiclass", "multilabel"]] = None,
    num_classes: Optional[int] = None,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    r"""Hinge loss.

    .. note::
        From v0.10 an ``'binary_*'``, ``'multiclass_*'``, ``'multilabel_*'`` version now exist of each classification
        metric. Moving forward we recommend using these versions. This base metric will still work as it did
        prior to v0.10 until v0.11. From v0.11 the `task` argument introduced in this metric will be required
        and the general order of arguments may change, such that this metric will just function as an single
        entrypoint to calling the three specialized versions.

    Computes the mean `Hinge loss`_ typically used for Support Vector Machines (SVMs).

     In the binary case it is defined as:

    .. math::
        \text{Hinge loss} = \max(0, 1 - y \times \hat{y})

    Where :math:`y \in {-1, 1}` is the target, and :math:`\hat{y} \in \mathbb{R}` is the prediction.

    In the multi-class case, when ``multiclass_mode=None`` (default), ``multiclass_mode=MulticlassMode.CRAMMER_SINGER``
    or ``multiclass_mode="crammer-singer"``, this metric will compute the multi-class hinge loss defined by Crammer and
    Singer as:

    .. math::
        \text{Hinge loss} = \max\left(0, 1 - \hat{y}_y + \max_{i \ne y} (\hat{y}_i)\right)

    Where :math:`y \in {0, ..., \mathrm{C}}` is the target class (where :math:`\mathrm{C}` is the number of classes),
    and :math:`\hat{y} \in \mathbb{R}^\mathrm{C}` is the predicted output per class.

    In the multi-class case when ``multiclass_mode=MulticlassMode.ONE_VS_ALL`` or ``multiclass_mode='one-vs-all'``, this
    metric will use a one-vs-all approach to compute the hinge loss, giving a vector of C outputs where each entry pits
    that class against all remaining classes.

    This metric can optionally output the mean of the squared hinge loss by setting ``squared=True``

    Only accepts inputs with preds shape of (N) (binary) or (N, C) (multi-class) and target shape of (N).

    Args:
        preds: Predictions from model (as float outputs from decision function).
        target: Ground truth labels.
        squared:
            If True, this will compute the squared hinge loss. Otherwise, computes the regular hinge loss (default).
        multiclass_mode:
            Which approach to use for multi-class inputs (has no effect in the binary case). ``None`` (default),
            ``MulticlassMode.CRAMMER_SINGER`` or ``"crammer-singer"``, uses the Crammer Singer multi-class hinge loss.
            ``MulticlassMode.ONE_VS_ALL`` or ``"one-vs-all"`` computes the hinge loss in a one-vs-all fashion.

    Raises:
        ValueError:
            If preds shape is not of size (N) or (N, C).
        ValueError:
            If target shape is not of size (N).
        ValueError:
            If ``multiclass_mode`` is not: None, ``MulticlassMode.CRAMMER_SINGER``, ``"crammer-singer"``,
            ``MulticlassMode.ONE_VS_ALL`` or ``"one-vs-all"``.

    Example (binary case):
        >>> import torch
        >>> from torchmetrics.functional import hinge_loss
        >>> target = torch.tensor([0, 1, 1])
        >>> preds = torch.tensor([-2.2, 2.4, 0.1])
        >>> hinge_loss(preds, target)
        tensor(0.3000)

    Example (default / multiclass case):
        >>> target = torch.tensor([0, 1, 2])
        >>> preds = torch.tensor([[-1.0, 0.9, 0.2], [0.5, -1.1, 0.8], [2.2, -0.5, 0.3]])
        >>> hinge_loss(preds, target)
        tensor(2.9000)

    Example (multiclass example, one vs all mode):
        >>> target = torch.tensor([0, 1, 2])
        >>> preds = torch.tensor([[-1.0, 0.9, 0.2], [0.5, -1.1, 0.8], [2.2, -0.5, 0.3]])
        >>> hinge_loss(preds, target, multiclass_mode="one-vs-all")
        tensor([2.2333, 1.5000, 1.2333])
    """
    if task is not None:
        if task == "binary":
            return binary_hinge_loss(preds, target, squared, ignore_index, validate_args)
        if task == "multiclass":
            assert isinstance(num_classes, int)
            assert multiclass_mode is not None
            return multiclass_hinge_loss(
                preds, target, num_classes, squared, multiclass_mode, ignore_index, validate_args
            )
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
    measure, total = _hinge_update(preds, target, squared=squared, multiclass_mode=multiclass_mode)
    return _hinge_compute(measure, total)
