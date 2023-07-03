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
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.classification.precision_recall_curve import (
    _binary_clf_curve,
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
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.compute import _safe_divide
from torchmetrics.utilities.enums import ClassificationTask


def _binary_roc_compute(
    state: Union[Tensor, Tuple[Tensor, Tensor]],
    thresholds: Optional[Tensor],
    pos_label: int = 1,
) -> Tuple[Tensor, Tensor, Tensor]:
    if isinstance(state, Tensor) and thresholds is not None:
        tps = state[:, 1, 1]
        fps = state[:, 0, 1]
        fns = state[:, 1, 0]
        tns = state[:, 0, 0]
        tpr = _safe_divide(tps, tps + fns).flip(0)
        fpr = _safe_divide(fps, fps + tns).flip(0)
        thres = thresholds.flip(0)
    else:
        fps, tps, thres = _binary_clf_curve(preds=state[0], target=state[1], pos_label=pos_label)
        # Add an extra threshold position to make sure that the curve starts at (0, 0)
        tps = torch.cat([torch.zeros(1, dtype=tps.dtype, device=tps.device), tps])
        fps = torch.cat([torch.zeros(1, dtype=fps.dtype, device=fps.device), fps])
        thres = torch.cat([torch.ones(1, dtype=thres.dtype, device=thres.device), thres])

        if fps[-1] <= 0:
            rank_zero_warn(
                "No negative samples in targets, false positive value should be meaningless."
                " Returning zero tensor in false positive score",
                UserWarning,
            )
            fpr = torch.zeros_like(thres)
        else:
            fpr = fps / fps[-1]

        if tps[-1] <= 0:
            rank_zero_warn(
                "No positive samples in targets, true positive value should be meaningless."
                " Returning zero tensor in true positive score",
                UserWarning,
            )
            tpr = torch.zeros_like(thres)
        else:
            tpr = tps / tps[-1]

    return fpr, tpr, thres


def binary_roc(
    preds: Tensor,
    target: Tensor,
    thresholds: Optional[Union[int, List[float], Tensor]] = None,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""Compute the Receiver Operating Characteristic (ROC) for binary tasks.

    The curve consist of multiple pairs of true positive rate (TPR) and false positive rate (FPR) values evaluated at
    different thresholds, such that the tradeoff between the two values can be seen.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (int tensor): ``(N, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain {0,1} values (except if `ignore_index` is specified). The value 1 always encodes the positive class.

    Additional dimension ``...`` will be flattened into the batch dimension.

    The implementation both supports calculating the metric in a non-binned but accurate version and a binned version
    that is less accurate but more memory efficient. Setting the `thresholds` argument to `None` will activate the
    non-binned  version that uses memory of size :math:`\mathcal{O}(n_{samples})` whereas setting the `thresholds`
    argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
    size :math:`\mathcal{O}(n_{thresholds})` (constant memory).

    Note that outputted thresholds will be in reversed order to ensure that they corresponds to both fpr and tpr which
    are sorted in reversed order during their calculation, such that they are monotome increasing.

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        thresholds:
            Can be one of:

            - If set to `None`, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an `int` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an `list` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d `tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        (tuple): a tuple of 3 tensors containing:

        - fpr: an 1d tensor of size (n_thresholds+1, ) with false positive rate values
        - tpr: an 1d tensor of size (n_thresholds+1, ) with true positive rate values
        - thresholds: an 1d tensor of size (n_thresholds, ) with decreasing threshold values

    Example:
        >>> from torchmetrics.functional.classification import binary_roc
        >>> preds = torch.tensor([0, 0.5, 0.7, 0.8])
        >>> target = torch.tensor([0, 1, 1, 0])
        >>> binary_roc(preds, target, thresholds=None)  # doctest: +NORMALIZE_WHITESPACE
        (tensor([0.0000, 0.5000, 0.5000, 0.5000, 1.0000]),
         tensor([0.0000, 0.0000, 0.5000, 1.0000, 1.0000]),
         tensor([1.0000, 0.8000, 0.7000, 0.5000, 0.0000]))
        >>> binary_roc(preds, target, thresholds=5)  # doctest: +NORMALIZE_WHITESPACE
        (tensor([0.0000, 0.5000, 0.5000, 0.5000, 1.0000]),
         tensor([0., 0., 1., 1., 1.]),
         tensor([1.0000, 0.7500, 0.5000, 0.2500, 0.0000]))
    """
    if validate_args:
        _binary_precision_recall_curve_arg_validation(thresholds, ignore_index)
        _binary_precision_recall_curve_tensor_validation(preds, target, ignore_index)
    preds, target, thresholds = _binary_precision_recall_curve_format(preds, target, thresholds, ignore_index)
    state = _binary_precision_recall_curve_update(preds, target, thresholds)
    return _binary_roc_compute(state, thresholds)


def _multiclass_roc_compute(
    state: Union[Tensor, Tuple[Tensor, Tensor]],
    num_classes: int,
    thresholds: Optional[Tensor],
) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor]]]:
    if isinstance(state, Tensor) and thresholds is not None:
        tps = state[:, :, 1, 1]
        fps = state[:, :, 0, 1]
        fns = state[:, :, 1, 0]
        tns = state[:, :, 0, 0]
        tpr = _safe_divide(tps, tps + fns).flip(0).T
        fpr = _safe_divide(fps, fps + tns).flip(0).T
        thres = thresholds.flip(0)
    else:
        fpr, tpr, thres = [], [], []  # type: ignore[assignment]
        for i in range(num_classes):
            res = _binary_roc_compute((state[0][:, i], state[1]), thresholds=None, pos_label=i)
            fpr.append(res[0])
            tpr.append(res[1])
            thres.append(res[2])
    return fpr, tpr, thres


def multiclass_roc(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    thresholds: Optional[Union[int, List[float], Tensor]] = None,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor]]]:
    r"""Compute the Receiver Operating Characteristic (ROC) for multiclass tasks.

    The curve consist of multiple pairs of true positive rate (TPR) and false positive rate (FPR) values evaluated at
    different thresholds, such that the tradeoff between the two values can be seen.

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

    Note that outputted thresholds will be in reversed order to ensure that they corresponds to both fpr and tpr which
    are sorted in reversed order during their calculation, such that they are monotome increasing.

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_classes: Integer specifing the number of classes
        thresholds:
            Can be one of:

            - If set to `None`, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an `int` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an `list` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d `tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        (tuple): a tuple of either 3 tensors or 3 lists containing

        - fpr: if `thresholds=None` a list for each class is returned with an 1d tensor of size (n_thresholds+1, )
          with false positive rate values (length may differ between classes). If `thresholds` is set to something else,
          then a single 2d tensor of size (n_classes, n_thresholds+1) with false positive rate values is returned.
        - tpr: if `thresholds=None` a list for each class is returned with an 1d tensor of size (n_thresholds+1, )
          with true positive rate values (length may differ between classes). If `thresholds` is set to something else,
          then a single 2d tensor of size (n_classes, n_thresholds+1) with true positive rate values is returned.
        - thresholds: if `thresholds=None` a list for each class is returned with an 1d tensor of size (n_thresholds, )
          with decreasing threshold values (length may differ between classes). If `threshold` is set to something else,
          then a single 1d tensor of size (n_thresholds, ) is returned with shared threshold values for all classes.

    Example:
        >>> from torchmetrics.functional.classification import multiclass_roc
        >>> preds = torch.tensor([[0.75, 0.05, 0.05, 0.05, 0.05],
        ...                       [0.05, 0.75, 0.05, 0.05, 0.05],
        ...                       [0.05, 0.05, 0.75, 0.05, 0.05],
        ...                       [0.05, 0.05, 0.05, 0.75, 0.05]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> fpr, tpr, thresholds = multiclass_roc(
        ...    preds, target, num_classes=5, thresholds=None
        ... )
        >>> fpr  # doctest: +NORMALIZE_WHITESPACE
        [tensor([0., 0., 1.]), tensor([0., 0., 1.]), tensor([0.0000, 0.3333, 1.0000]),
         tensor([0.0000, 0.3333, 1.0000]), tensor([0., 1.])]
        >>> tpr
        [tensor([0., 1., 1.]), tensor([0., 1., 1.]), tensor([0., 0., 1.]), tensor([0., 0., 1.]), tensor([0., 0.])]
        >>> thresholds  # doctest: +NORMALIZE_WHITESPACE
        [tensor([1.0000, 0.7500, 0.0500]), tensor([1.0000, 0.7500, 0.0500]),
         tensor([1.0000, 0.7500, 0.0500]), tensor([1.0000, 0.7500, 0.0500]), tensor([1.0000, 0.0500])]
        >>> multiclass_roc(
        ...     preds, target, num_classes=5, thresholds=5
        ... )  # doctest: +NORMALIZE_WHITESPACE
        (tensor([[0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
                 [0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
                 [0.0000, 0.3333, 0.3333, 0.3333, 1.0000],
                 [0.0000, 0.3333, 0.3333, 0.3333, 1.0000],
                 [0.0000, 0.0000, 0.0000, 0.0000, 1.0000]]),
         tensor([[0., 1., 1., 1., 1.],
                 [0., 1., 1., 1., 1.],
                 [0., 0., 0., 0., 1.],
                 [0., 0., 0., 0., 1.],
                 [0., 0., 0., 0., 0.]]),
         tensor([1.0000, 0.7500, 0.5000, 0.2500, 0.0000]))
    """
    if validate_args:
        _multiclass_precision_recall_curve_arg_validation(num_classes, thresholds, ignore_index)
        _multiclass_precision_recall_curve_tensor_validation(preds, target, num_classes, ignore_index)
    preds, target, thresholds = _multiclass_precision_recall_curve_format(
        preds, target, num_classes, thresholds, ignore_index
    )
    state = _multiclass_precision_recall_curve_update(preds, target, num_classes, thresholds)
    return _multiclass_roc_compute(state, num_classes, thresholds)


def _multilabel_roc_compute(
    state: Union[Tensor, Tuple[Tensor, Tensor]],
    num_labels: int,
    thresholds: Optional[Tensor],
    ignore_index: Optional[int] = None,
) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor]]]:
    if isinstance(state, Tensor) and thresholds is not None:
        tps = state[:, :, 1, 1]
        fps = state[:, :, 0, 1]
        fns = state[:, :, 1, 0]
        tns = state[:, :, 0, 0]
        tpr = _safe_divide(tps, tps + fns).flip(0).T
        fpr = _safe_divide(fps, fps + tns).flip(0).T
        thres = thresholds.flip(0)
    else:
        fpr, tpr, thres = [], [], []  # type: ignore[assignment]
        for i in range(num_labels):
            preds = state[0][:, i]
            target = state[1][:, i]
            if ignore_index is not None:
                idx = target == ignore_index
                preds = preds[~idx]
                target = target[~idx]
            res = _binary_roc_compute((preds, target), thresholds=None, pos_label=1)
            fpr.append(res[0])
            tpr.append(res[1])
            thres.append(res[2])
    return fpr, tpr, thres


def multilabel_roc(
    preds: Tensor,
    target: Tensor,
    num_labels: int,
    thresholds: Optional[Union[int, List[float], Tensor]] = None,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor]]]:
    r"""Compute the Receiver Operating Characteristic (ROC) for multilabel tasks.

    The curve consist of multiple pairs of true positive rate (TPR) and false positive rate (FPR) values evaluated at
    different thresholds, such that the tradeoff between the two values can be seen.

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

    Note that outputted thresholds will be in reversed order to ensure that they corresponds to both fpr and tpr which
    are sorted in reversed order during their calculation, such that they are monotome increasing.

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_labels: Integer specifing the number of labels
        thresholds:
            Can be one of:

            - If set to `None`, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an `int` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an `list` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d `tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        (tuple): a tuple of either 3 tensors or 3 lists containing

        - fpr: if `thresholds=None` a list for each label is returned with an 1d tensor of size (n_thresholds+1, )
          with false positive rate values (length may differ between labels). If `thresholds` is set to something else,
          then a single 2d tensor of size (n_labels, n_thresholds+1) with false positive rate values is returned.
        - tpr: if `thresholds=None` a list for each label is returned with an 1d tensor of size (n_thresholds+1, )
          with true positive rate values (length may differ between labels). If `thresholds` is set to something else,
          then a single 2d tensor of size (n_labels, n_thresholds+1) with true positive rate values is returned.
        - thresholds: if `thresholds=None` a list for each label is returned with an 1d tensor of size (n_thresholds, )
          with decreasing threshold values (length may differ between labels). If `threshold` is set to something else,
          then a single 1d tensor of size (n_thresholds, ) is returned with shared threshold values for all labels.

    Example:
        >>> from torchmetrics.functional.classification import multilabel_roc
        >>> preds = torch.tensor([[0.75, 0.05, 0.35],
        ...                       [0.45, 0.75, 0.05],
        ...                       [0.05, 0.55, 0.75],
        ...                       [0.05, 0.65, 0.05]])
        >>> target = torch.tensor([[1, 0, 1],
        ...                        [0, 0, 0],
        ...                        [0, 1, 1],
        ...                        [1, 1, 1]])
        >>> fpr, tpr, thresholds = multilabel_roc(
        ...    preds, target, num_labels=3, thresholds=None
        ... )
        >>> fpr  # doctest: +NORMALIZE_WHITESPACE
        [tensor([0.0000, 0.0000, 0.5000, 1.0000]),
         tensor([0.0000, 0.5000, 0.5000, 0.5000, 1.0000]),
         tensor([0., 0., 0., 1.])]
        >>> tpr  # doctest: +NORMALIZE_WHITESPACE
        [tensor([0.0000, 0.5000, 0.5000, 1.0000]),
         tensor([0.0000, 0.0000, 0.5000, 1.0000, 1.0000]),
         tensor([0.0000, 0.3333, 0.6667, 1.0000])]
        >>> thresholds  # doctest: +NORMALIZE_WHITESPACE
        [tensor([1.0000, 0.7500, 0.4500, 0.0500]),
         tensor([1.0000, 0.7500, 0.6500, 0.5500, 0.0500]),
         tensor([1.0000, 0.7500, 0.3500, 0.0500])]
        >>> multilabel_roc(
        ...     preds, target, num_labels=3, thresholds=5
        ... )  # doctest: +NORMALIZE_WHITESPACE
        (tensor([[0.0000, 0.0000, 0.0000, 0.5000, 1.0000],
                 [0.0000, 0.5000, 0.5000, 0.5000, 1.0000],
                 [0.0000, 0.0000, 0.0000, 0.0000, 1.0000]]),
         tensor([[0.0000, 0.5000, 0.5000, 0.5000, 1.0000],
                 [0.0000, 0.0000, 1.0000, 1.0000, 1.0000],
                 [0.0000, 0.3333, 0.3333, 0.6667, 1.0000]]),
         tensor([1.0000, 0.7500, 0.5000, 0.2500, 0.0000]))
    """
    if validate_args:
        _multilabel_precision_recall_curve_arg_validation(num_labels, thresholds, ignore_index)
        _multilabel_precision_recall_curve_tensor_validation(preds, target, num_labels, ignore_index)
    preds, target, thresholds = _multilabel_precision_recall_curve_format(
        preds, target, num_labels, thresholds, ignore_index
    )
    state = _multilabel_precision_recall_curve_update(preds, target, num_labels, thresholds)
    return _multilabel_roc_compute(state, num_labels, thresholds, ignore_index)


def roc(
    preds: Tensor,
    target: Tensor,
    task: Literal["binary", "multiclass", "multilabel"],
    thresholds: Optional[Union[int, List[float], Tensor]] = None,
    num_classes: Optional[int] = None,
    num_labels: Optional[int] = None,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor]]]:
    r"""Compute the Receiver Operating Characteristic (ROC).

    The curve consist of multiple pairs of true positive rate (TPR) and false positive rate (FPR) values evaluated at
    different thresholds, such that the tradeoff between the two values can be seen.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``multilabel``. See the documentation of
    :func:`binary_roc`, :func:`multiclass_roc` and :func:`multilabel_roc` for the specific details of each argument
    influence and examples.

    Legacy Example:
        >>> pred = torch.tensor([0.0, 1.0, 2.0, 3.0])
        >>> target = torch.tensor([0, 1, 1, 1])
        >>> fpr, tpr, thresholds = roc(pred, target, task='binary')
        >>> fpr
        tensor([0., 0., 0., 0., 1.])
        >>> tpr
        tensor([0.0000, 0.3333, 0.6667, 1.0000, 1.0000])
        >>> thresholds
        tensor([1.0000, 0.9526, 0.8808, 0.7311, 0.5000])

        >>> pred = torch.tensor([[0.75, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.75, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.75, 0.05],
        ...                      [0.05, 0.05, 0.05, 0.75]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> fpr, tpr, thresholds = roc(pred, target, task='multiclass', num_classes=4)
        >>> fpr
        [tensor([0., 0., 1.]), tensor([0., 0., 1.]), tensor([0.0000, 0.3333, 1.0000]), tensor([0.0000, 0.3333, 1.0000])]
        >>> tpr
        [tensor([0., 1., 1.]), tensor([0., 1., 1.]), tensor([0., 0., 1.]), tensor([0., 0., 1.])]
        >>> thresholds
        [tensor([1.0000, 0.7500, 0.0500]),
         tensor([1.0000, 0.7500, 0.0500]),
         tensor([1.0000, 0.7500, 0.0500]),
         tensor([1.0000, 0.7500, 0.0500])]

        >>> pred = torch.tensor([[0.8191, 0.3680, 0.1138],
        ...                      [0.3584, 0.7576, 0.1183],
        ...                      [0.2286, 0.3468, 0.1338],
        ...                      [0.8603, 0.0745, 0.1837]])
        >>> target = torch.tensor([[1, 1, 0], [0, 1, 0], [0, 0, 0], [0, 1, 1]])
        >>> fpr, tpr, thresholds = roc(pred, target, task='multilabel', num_labels=3)
        >>> fpr
        [tensor([0.0000, 0.3333, 0.3333, 0.6667, 1.0000]),
         tensor([0., 0., 0., 1., 1.]),
         tensor([0.0000, 0.0000, 0.3333, 0.6667, 1.0000])]
        >>> tpr
        [tensor([0., 0., 1., 1., 1.]), tensor([0.0000, 0.3333, 0.6667, 0.6667, 1.0000]), tensor([0., 1., 1., 1., 1.])]
        >>> thresholds
        [tensor([1.0000, 0.8603, 0.8191, 0.3584, 0.2286]),
         tensor([1.0000, 0.7576, 0.3680, 0.3468, 0.0745]),
         tensor([1.0000, 0.1837, 0.1338, 0.1183, 0.1138])]
    """
    task = ClassificationTask.from_str(task)
    if task == ClassificationTask.BINARY:
        return binary_roc(preds, target, thresholds, ignore_index, validate_args)
    if task == ClassificationTask.MULTICLASS:
        if not isinstance(num_classes, int):
            raise ValueError(f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`")
        return multiclass_roc(preds, target, num_classes, thresholds, ignore_index, validate_args)
    if task == ClassificationTask.MULTILABEL:
        if not isinstance(num_labels, int):
            raise ValueError(f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`")
        return multilabel_roc(preds, target, num_labels, thresholds, ignore_index, validate_args)
    raise ValueError(f"Task {task} not supported, expected one of {ClassificationTask}.")
