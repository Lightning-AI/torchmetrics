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
from typing import List, Optional, Union

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.classification.precision_recall_curve import (
    _binary_precision_recall_curve_arg_validation,
    _binary_precision_recall_curve_format,
    _binary_precision_recall_curve_tensor_validation,
    _binary_precision_recall_curve_update,
    _multiclass_precision_recall_curve_arg_validation,
    _multiclass_precision_recall_curve_format,
    _multiclass_precision_recall_curve_tensor_validation,
    _multiclass_precision_recall_curve_update,
    _multilabel_precision_recall_curve_arg_validation,
    _multilabel_precision_recall_curve_format,
    _multilabel_precision_recall_curve_tensor_validation,
    _multilabel_precision_recall_curve_update,
)
from torchmetrics.functional.classification.roc import (
    _binary_roc_compute,
    _multiclass_roc_compute,
    _multilabel_roc_compute,
)
from torchmetrics.utilities.enums import ClassificationTask


def _convert_fpr_to_specificity(fpr: Tensor) -> Tensor:
    """Convert  fprs to specificity."""
    return 1 - fpr


def _sensitivity_at_specificity(
    sensitivity: Tensor,
    specificity: Tensor,
    thresholds: Tensor,
    min_specificity: float,
) -> tuple[Tensor, Tensor]:
    # get indices where specificity is greater than min_specificity
    indices = specificity >= min_specificity

    # if no indices are found, max_spec, best_threshold = 0.0, 1e6
    if not indices.any():
        max_spec = torch.tensor(0.0, device=sensitivity.device, dtype=sensitivity.dtype)
        best_threshold = torch.tensor(1e6, device=thresholds.device, dtype=thresholds.dtype)
    else:
        # redefine sensitivity, specificity and threshold tensor based on indices
        sensitivity, specificity, thresholds = sensitivity[indices], specificity[indices], thresholds[indices]

        # get argmax
        idx = torch.argmax(sensitivity)

        # get max_spec and best_threshold
        max_spec, best_threshold = sensitivity[idx], thresholds[idx]

    return max_spec, best_threshold


def _binary_sensitivity_at_specificity_arg_validation(
    min_specificity: float,
    thresholds: Optional[Union[int, list[float], Tensor]] = None,
    ignore_index: Optional[int] = None,
) -> None:
    _binary_precision_recall_curve_arg_validation(thresholds, ignore_index)
    if not isinstance(min_specificity, float) and not (0 <= min_specificity <= 1):
        raise ValueError(
            f"Expected argument `min_specificity` to be an float in the [0,1] range, but got {min_specificity}"
        )


def _binary_sensitivity_at_specificity_compute(
    state: Union[Tensor, tuple[Tensor, Tensor]],
    thresholds: Optional[Tensor],
    min_specificity: float,
    pos_label: int = 1,
) -> tuple[Tensor, Tensor]:
    fpr, sensitivity, thresholds = _binary_roc_compute(state, thresholds, pos_label)
    specificity = _convert_fpr_to_specificity(fpr)
    return _sensitivity_at_specificity(sensitivity, specificity, thresholds, min_specificity)


def binary_sensitivity_at_specificity(
    preds: Tensor,
    target: Tensor,
    min_specificity: float,
    thresholds: Optional[Union[int, list[float], Tensor]] = None,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> tuple[Tensor, Tensor]:
    r"""Compute the highest possible sensitivity value given the minimum specificity levels provided for binary tasks.

    This is done by first calculating the Receiver Operating Characteristic (ROC) curve for different thresholds and
    the find the sensitivity for a given specificity level.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (int tensor): ``(N, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain {0,1} values (except if `ignore_index` is specified).

    Additional dimension ``...`` will be flattened into the batch dimension.

    The implementation both supports calculating the metric in a non-binned but accurate version and a binned version
    that is less accurate but more memory efficient. Setting the `thresholds` argument to `None` will activate the
    non-binned  version that uses memory of size :math:`\mathcal{O}(n_{samples})` whereas setting the `thresholds`
    argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
    size :math:`\mathcal{O}(n_{thresholds})` (constant memory).

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        min_specificity: float value specifying minimum specificity threshold.
        thresholds:
            Can be one of:

            - ``None``, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. It is the most accurate but also the most memory-consuming approach.
            - ``int`` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - ``list`` of floats, will use the indicated thresholds in the list as bins for the calculation
            - 1d ``tensor`` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        (tuple): a tuple of 2 tensors containing:

        - sensitivity: a scalar tensor with the maximum sensitivity for the given specificity level
        - threshold: a scalar tensor with the corresponding threshold level

    Example:
        >>> from torchmetrics.functional.classification import binary_sensitivity_at_specificity
        >>> preds = torch.tensor([0, 0.5, 0.4, 0.1])
        >>> target = torch.tensor([0, 1, 1, 1])
        >>> binary_sensitivity_at_specificity(preds, target, min_specificity=0.5, thresholds=None)
        (tensor(1.), tensor(0.1000))
        >>> binary_sensitivity_at_specificity(preds, target, min_specificity=0.5, thresholds=5)
        (tensor(0.6667), tensor(0.2500))

    """
    if validate_args:
        _binary_sensitivity_at_specificity_arg_validation(min_specificity, thresholds, ignore_index)
        _binary_precision_recall_curve_tensor_validation(preds, target, ignore_index)
    preds, target, thresholds = _binary_precision_recall_curve_format(preds, target, thresholds, ignore_index)
    state = _binary_precision_recall_curve_update(preds, target, thresholds)
    return _binary_sensitivity_at_specificity_compute(state, thresholds, min_specificity)


def _multiclass_sensitivity_at_specificity_arg_validation(
    num_classes: int,
    min_specificity: float,
    thresholds: Optional[Union[int, list[float], Tensor]] = None,
    ignore_index: Optional[int] = None,
) -> None:
    _multiclass_precision_recall_curve_arg_validation(num_classes, thresholds, ignore_index)
    if not isinstance(min_specificity, float) and not (0 <= min_specificity <= 1):
        raise ValueError(
            f"Expected argument `min_specificity` to be an float in the [0,1] range, but got {min_specificity}"
        )


def _multiclass_sensitivity_at_specificity_compute(
    state: Union[Tensor, tuple[Tensor, Tensor]],
    num_classes: int,
    thresholds: Optional[Tensor],
    min_specificity: float,
) -> tuple[Tensor, Tensor]:
    fpr, sensitivity, thresholds = _multiclass_roc_compute(state, num_classes, thresholds)
    specificity = [_convert_fpr_to_specificity(fpr_) for fpr_ in fpr]
    if isinstance(state, Tensor):
        res = [
            _sensitivity_at_specificity(sp, sn, thresholds, min_specificity)  # type: ignore
            for sp, sn in zip(sensitivity, specificity)
        ]
    else:
        res = [
            _sensitivity_at_specificity(sp, sn, t, min_specificity)
            for sp, sn, t in zip(sensitivity, specificity, thresholds)
        ]
    sensitivity = torch.stack([r[0] for r in res])
    thresholds = torch.stack([r[1] for r in res])
    return sensitivity, thresholds


def multiclass_sensitivity_at_specificity(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    min_specificity: float,
    thresholds: Optional[Union[int, list[float], Tensor]] = None,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> tuple[Tensor, Tensor]:
    r"""Compute the highest possible sensitivity value given minimum specificity level provided for multiclass tasks.

    This is done by first calculating the Receiver Operating Characteristic (ROC) curve for different thresholds and the
    find the sensitivity for a given specificity level.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, C, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      softmax per sample.
    - ``target`` (int tensor): ``(N, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain values in the [0, n_classes-1] range (except if `ignore_index` is specified).

    Additional dimension ``...`` will be flattened into the batch dimension.

    The implementation both supports calculating the metric in a non-binned but accurate version and a binned version
    that is less accurate but more memory efficient. Setting the `thresholds` argument to `None` will activate the
    non-binned  version that uses memory of size :math:`\mathcal{O}(n_{samples})` whereas setting the `thresholds`
    argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
    size :math:`\mathcal{O}(n_{thresholds} \times n_{classes})` (constant memory).

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_classes: Integer specifying the number of classes
        min_specificity: float value specifying minimum specificity threshold.
        thresholds:
            Can be one of:

            - ``None``, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. It is the most accurate but also the most memory-consuming approach.
            - ``int`` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - ``list`` of floats, will use the indicated thresholds in the list as bins for the calculation
            - 1d ``tensor`` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        (tuple): a tuple of either 2 tensors or 2 lists containing

        - recall: an 1d tensor of size ``(n_classes, )`` with the maximum recall for the given precision level per class
        - thresholds: an 1d tensor of size ``(n_classes, )`` with the corresponding threshold level per class

    Example:
        >>> from torchmetrics.functional.classification import multiclass_sensitivity_at_specificity
        >>> preds = torch.tensor([[0.75, 0.05, 0.05, 0.05, 0.05],
        ...                       [0.05, 0.75, 0.05, 0.05, 0.05],
        ...                       [0.05, 0.05, 0.75, 0.05, 0.05],
        ...                       [0.05, 0.05, 0.05, 0.75, 0.05]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> multiclass_sensitivity_at_specificity(preds, target, num_classes=5, min_specificity=0.5, thresholds=None)
        (tensor([1., 1., 0., 0., 0.]), tensor([0.7500, 0.7500, 1.0000, 1.0000, 1.0000]))
        >>> multiclass_sensitivity_at_specificity(preds, target, num_classes=5, min_specificity=0.5, thresholds=5)
        (tensor([1., 1., 0., 0., 0.]), tensor([0.7500, 0.7500, 1.0000, 1.0000, 1.0000]))

    """
    if validate_args:
        _multiclass_sensitivity_at_specificity_arg_validation(num_classes, min_specificity, thresholds, ignore_index)
        _multiclass_precision_recall_curve_tensor_validation(preds, target, num_classes, ignore_index)
    preds, target, thresholds = _multiclass_precision_recall_curve_format(
        preds, target, num_classes, thresholds, ignore_index
    )
    state = _multiclass_precision_recall_curve_update(preds, target, num_classes, thresholds)
    return _multiclass_sensitivity_at_specificity_compute(state, num_classes, thresholds, min_specificity)


def _multilabel_sensitivity_at_specificity_arg_validation(
    num_labels: int,
    min_specificity: float,
    thresholds: Optional[Union[int, list[float], Tensor]] = None,
    ignore_index: Optional[int] = None,
) -> None:
    _multilabel_precision_recall_curve_arg_validation(num_labels, thresholds, ignore_index)
    if not isinstance(min_specificity, float) and not (0 <= min_specificity <= 1):
        raise ValueError(
            f"Expected argument `min_specificity` to be an float in the [0,1] range, but got {min_specificity}"
        )


def _multilabel_sensitivity_at_specificity_compute(
    state: Union[Tensor, tuple[Tensor, Tensor]],
    num_labels: int,
    thresholds: Optional[Tensor],
    ignore_index: Optional[int],
    min_specificity: float,
) -> tuple[Tensor, Tensor]:
    fpr, sensitivity, thresholds = _multilabel_roc_compute(state, num_labels, thresholds, ignore_index)
    specificity = [_convert_fpr_to_specificity(fpr_) for fpr_ in fpr]
    if isinstance(state, Tensor):
        res = [
            _sensitivity_at_specificity(sp, sn, thresholds, min_specificity)  # type: ignore
            for sp, sn in zip(sensitivity, specificity)
        ]
    else:
        res = [
            _sensitivity_at_specificity(sp, sn, t, min_specificity)
            for sp, sn, t in zip(sensitivity, specificity, thresholds)
        ]
    sensitivity = torch.stack([r[0] for r in res])
    thresholds = torch.stack([r[1] for r in res])
    return sensitivity, thresholds


def multilabel_sensitivity_at_specificity(
    preds: Tensor,
    target: Tensor,
    num_labels: int,
    min_specificity: float,
    thresholds: Optional[Union[int, list[float], Tensor]] = None,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> tuple[Tensor, Tensor]:
    r"""Compute the highest possible sensitivity value given minimum specificity level provided for multilabel tasks.

    This is done by first calculating the Receiver Operating Characteristic (ROC) curve for different thresholds and
    the find the sensitivity for a given specificity level.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, C, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (int tensor): ``(N, C, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain {0,1} values (except if `ignore_index` is specified).

    Additional dimension ``...`` will be flattened into the batch dimension.

    The implementation both supports calculating the metric in a non-binned but accurate version and a binned version
    that is less accurate but more memory efficient. Setting the `thresholds` argument to `None` will activate the
    non-binned  version that uses memory of size :math:`\mathcal{O}(n_{samples})` whereas setting the `thresholds`
    argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
    size :math:`\mathcal{O}(n_{thresholds} \times n_{labels})` (constant memory).

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_labels: Integer specifying the number of labels
        min_specificity: float value specifying minimum specificity threshold.
        thresholds:
            Can be one of:

            - ``None``, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. It is the most accurate but also the most memory-consuming approach.
            - ``int`` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - ``list`` of floats, will use the indicated thresholds in the list as bins for the calculation
            - 1d ``tensor`` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        (tuple): a tuple of either 2 tensors or 2 lists containing

        - sensitivity: an 1d tensor of size (n_classes, ) with the maximum recall for the given precision
            level per class
        - thresholds: an 1d tensor of size (n_classes, ) with the corresponding threshold level per class

    Example:
        >>> from torchmetrics.functional.classification import multilabel_sensitivity_at_specificity
        >>> preds = torch.tensor([[0.75, 0.05, 0.35],
        ...                       [0.45, 0.75, 0.05],
        ...                       [0.05, 0.55, 0.75],
        ...                       [0.05, 0.65, 0.05]])
        >>> target = torch.tensor([[1, 0, 1],
        ...                        [0, 0, 0],
        ...                        [0, 1, 1],
        ...                        [1, 1, 1]])
        >>> multilabel_sensitivity_at_specificity(preds, target, num_labels=3, min_specificity=0.5, thresholds=None)
        (tensor([0.5000, 1.0000, 0.6667]), tensor([0.7500, 0.5500, 0.3500]))
        >>> multilabel_sensitivity_at_specificity(preds, target, num_labels=3, min_specificity=0.5, thresholds=5)
        (tensor([0.5000, 1.0000, 0.6667]), tensor([0.7500, 0.5000, 0.2500]))

    """
    if validate_args:
        _multilabel_sensitivity_at_specificity_arg_validation(num_labels, min_specificity, thresholds, ignore_index)
        _multilabel_precision_recall_curve_tensor_validation(preds, target, num_labels, ignore_index)
    preds, target, thresholds = _multilabel_precision_recall_curve_format(
        preds, target, num_labels, thresholds, ignore_index
    )
    state = _multilabel_precision_recall_curve_update(preds, target, num_labels, thresholds)
    return _multilabel_sensitivity_at_specificity_compute(state, num_labels, thresholds, ignore_index, min_specificity)


def sensitivity_at_specificity(
    preds: Tensor,
    target: Tensor,
    task: Literal["binary", "multiclass", "multilabel"],
    min_specificity: float,
    thresholds: Optional[Union[int, list[float], Tensor]] = None,
    num_classes: Optional[int] = None,
    num_labels: Optional[int] = None,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Union[Tensor, tuple[Tensor, Tensor, Tensor], tuple[List[Tensor], List[Tensor], List[Tensor]]]:
    r"""Compute the highest possible sensitivity value given the minimum specificity thresholds provided.

    This is done by first calculating the Receiver Operating Characteristic (ROC) curve for different thresholds and
    the find the sensitivity for a given specificity level.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``'multilabel'``. See the documentation of
    :func:`~torchmetrics.functional.classification.binary_sensitivity_at_specificity`,
    :func:`~torchmetrics.functional.classification.multiclass_sensitivity_at_specificity` and
    :func:`~torchmetrics.functional.classification.multilabel_sensitivity_at_specificity` for the specific details of
    each argument influence and examples.

    """
    task = ClassificationTask.from_str(task)
    if task == ClassificationTask.BINARY:
        return binary_sensitivity_at_specificity(  # type: ignore
            preds, target, min_specificity, thresholds, ignore_index, validate_args
        )
    if task == ClassificationTask.MULTICLASS:
        if not isinstance(num_classes, int):
            raise ValueError(f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`")
        return multiclass_sensitivity_at_specificity(  # type: ignore
            preds, target, num_classes, min_specificity, thresholds, ignore_index, validate_args
        )
    if task == ClassificationTask.MULTILABEL:
        if not isinstance(num_labels, int):
            raise ValueError(f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`")
        return multilabel_sensitivity_at_specificity(  # type: ignore
            preds, target, num_labels, min_specificity, thresholds, ignore_index, validate_args
        )
    raise ValueError(f"Not handled value: {task}")
