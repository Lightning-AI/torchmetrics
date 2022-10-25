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
from typing_extensions import Literal

from torchmetrics.functional.classification.stat_scores import (
    _binary_stat_scores_arg_validation,
    _binary_stat_scores_format,
    _binary_stat_scores_tensor_validation,
    _binary_stat_scores_update,
    _multiclass_stat_scores_arg_validation,
    _multiclass_stat_scores_format,
    _multiclass_stat_scores_tensor_validation,
    _multiclass_stat_scores_update,
    _multilabel_stat_scores_arg_validation,
    _multilabel_stat_scores_format,
    _multilabel_stat_scores_tensor_validation,
    _multilabel_stat_scores_update,
    _reduce_stat_scores,
    _stat_scores_update,
)
from torchmetrics.utilities.checks import _check_classification_inputs, _input_format_classification, _input_squeeze
from torchmetrics.utilities.compute import _safe_divide
from torchmetrics.utilities.enums import AverageMethod, DataType, MDMCAverageMethod
from torchmetrics.utilities.prints import rank_zero_warn


def _accuracy_reduce(
    tp: Tensor,
    fp: Tensor,
    tn: Tensor,
    fn: Tensor,
    average: Optional[Literal["binary", "micro", "macro", "weighted", "none"]],
    multidim_average: Literal["global", "samplewise"] = "global",
    multilabel: bool = False,
) -> Tensor:
    """Reduce classification statistics into accuracy score
    Args:
        tp: number of true positives
        fp: number of false positives
        tn: number of true negatives
        fn: number of false negatives
        normalize: normalization method.
            - `"true"` will divide by the sum of the column dimension.
            - `"pred"` will divide by the sum of the row dimension.
            - `"all"` will divide by the sum of the full matrix
            - `"none"` or `None` will apply no reduction
        multilabel: bool indicating if reduction is for multilabel tasks

    Returns:
        Accuracy score
    """
    if average == "binary":
        return _safe_divide(tp + tn, tp + tn + fp + fn)
    elif average == "micro":
        tp = tp.sum(dim=0 if multidim_average == "global" else 1)
        fn = fn.sum(dim=0 if multidim_average == "global" else 1)
        if multilabel:
            fp = fp.sum(dim=0 if multidim_average == "global" else 1)
            tn = tn.sum(dim=0 if multidim_average == "global" else 1)
            return _safe_divide(tp + tn, tp + tn + fp + fn)
        return _safe_divide(tp, tp + fn)
    else:
        if multilabel:
            score = _safe_divide(tp + tn, tp + tn + fp + fn)
        else:
            score = _safe_divide(tp, tp + fn)
        if average is None or average == "none":
            return score
        if average == "weighted":
            weights = tp + fn
        else:
            weights = torch.ones_like(score)
        return _safe_divide(weights * score, weights.sum(-1, keepdim=True)).sum(-1)


def binary_accuracy(
    preds: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    r"""Computes `Accuracy`_ for binary tasks:

    .. math::
        \text{Accuracy} = \frac{1}{N}\sum_i^N 1(y_i = \hat{y}_i)

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a
    tensor of predictions.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Addtionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, ...)``

    The influence of the additional dimension ``...`` (if present) will be determined by the `multidim_average`
    argument.

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        threshold: Threshold for transforming probability to binary {0,1} predictions
        multidim_average:
            Defines how additionally dimensions ``...`` should be handled. Should be one of the following:

            - ``global``: Additional dimensions are flatted along the batch dimension
            - ``samplewise``: Statistic will be calculated independently for each sample on the ``N`` axis.
              The statistics in this case are calculated over the additional dimensions.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        If ``multidim_average`` is set to ``global``, the metric returns a scalar value. If ``multidim_average``
        is set to ``samplewise``, the metric returns ``(N,)`` vector consisting of a scalar value per sample.

    Example (preds is int tensor):
        >>> from torchmetrics.functional.classification import binary_accuracy
        >>> target = torch.tensor([0, 1, 0, 1, 0, 1])
        >>> preds = torch.tensor([0, 0, 1, 1, 0, 1])
        >>> binary_accuracy(preds, target)
        tensor(0.6667)

    Example (preds is float tensor):
        >>> from torchmetrics.functional.classification import binary_accuracy
        >>> target = torch.tensor([0, 1, 0, 1, 0, 1])
        >>> preds = torch.tensor([0.11, 0.22, 0.84, 0.73, 0.33, 0.92])
        >>> binary_accuracy(preds, target)
        tensor(0.6667)

    Example (multidim tensors):
        >>> from torchmetrics.functional.classification import binary_accuracy
        >>> target = torch.tensor([[[0, 1], [1, 0], [0, 1]], [[1, 1], [0, 0], [1, 0]]])
        >>> preds = torch.tensor(
        ...     [
        ...         [[0.59, 0.91], [0.91, 0.99], [0.63, 0.04]],
        ...         [[0.38, 0.04], [0.86, 0.780], [0.45, 0.37]],
        ...     ]
        ... )
        >>> binary_accuracy(preds, target, multidim_average='samplewise')
        tensor([0.3333, 0.1667])
    """
    if validate_args:
        _binary_stat_scores_arg_validation(threshold, multidim_average, ignore_index)
        _binary_stat_scores_tensor_validation(preds, target, multidim_average, ignore_index)
    preds, target = _binary_stat_scores_format(preds, target, threshold, ignore_index)
    tp, fp, tn, fn = _binary_stat_scores_update(preds, target, multidim_average)
    return _accuracy_reduce(tp, fp, tn, fn, average="binary", multidim_average=multidim_average)


def multiclass_accuracy(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
    top_k: int = 1,
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    r"""Computes `Accuracy`_ for multiclass tasks:

    .. math::
        \text{Accuracy} = \frac{1}{N}\sum_i^N 1(y_i = \hat{y}_i)

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a
    tensor of predictions.

    Accepts the following input tensors:

    - ``preds``: ``(N, ...)`` (int tensor) or ``(N, C, ..)`` (float tensor). If preds is a floating point
      we apply ``torch.argmax`` along the ``C`` dimension to automatically convert probabilities/logits into
      an int tensor.
    - ``target`` (int tensor): ``(N, ...)``

    The influence of the additional dimension ``...`` (if present) will be determined by the `multidim_average`
    argument.

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_classes: Integer specifing the number of classes
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: Calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: Calculates statistic for each label and applies no reduction

        top_k:
            Number of highest probability or logit score predictions considered to find the correct label.
            Only works when ``preds`` contain probabilities/logits.
        multidim_average:
            Defines how additionally dimensions ``...`` should be handled. Should be one of the following:

            - ``global``: Additional dimensions are flatted along the batch dimension
            - ``samplewise``: Statistic will be calculated independently for each sample on the ``N`` axis.
              The statistics in this case are calculated over the additional dimensions.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        The returned shape depends on the ``average`` and ``multidim_average`` arguments:

        - If ``multidim_average`` is set to ``global``:

          - If ``average='micro'/'macro'/'weighted'``, the output will be a scalar tensor
          - If ``average=None/'none'``, the shape will be ``(C,)``

        - If ``multidim_average`` is set to ``samplewise``:

          - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(N,)``
          - If ``average=None/'none'``, the shape will be ``(N, C)``

    Example (preds is int tensor):
        >>> from torchmetrics.functional.classification import multiclass_accuracy
        >>> target = torch.tensor([2, 1, 0, 0])
        >>> preds = torch.tensor([2, 1, 0, 1])
        >>> multiclass_accuracy(preds, target, num_classes=3)
        tensor(0.8333)
        >>> multiclass_accuracy(preds, target, num_classes=3, average=None)
        tensor([0.5000, 1.0000, 1.0000])

    Example (preds is float tensor):
        >>> from torchmetrics.functional.classification import multiclass_accuracy
        >>> target = torch.tensor([2, 1, 0, 0])
        >>> preds = torch.tensor([
        ...   [0.16, 0.26, 0.58],
        ...   [0.22, 0.61, 0.17],
        ...   [0.71, 0.09, 0.20],
        ...   [0.05, 0.82, 0.13],
        ... ])
        >>> multiclass_accuracy(preds, target, num_classes=3)
        tensor(0.8333)
        >>> multiclass_accuracy(preds, target, num_classes=3, average=None)
        tensor([0.5000, 1.0000, 1.0000])

    Example (multidim tensors):
        >>> from torchmetrics.functional.classification import multiclass_accuracy
        >>> target = torch.tensor([[[0, 1], [2, 1], [0, 2]], [[1, 1], [2, 0], [1, 2]]])
        >>> preds = torch.tensor([[[0, 2], [2, 0], [0, 1]], [[2, 2], [2, 1], [1, 0]]])
        >>> multiclass_accuracy(preds, target, num_classes=3, multidim_average='samplewise')
        tensor([0.5000, 0.2778])
        >>> multiclass_accuracy(preds, target, num_classes=3, multidim_average='samplewise', average=None)
        tensor([[1.0000, 0.0000, 0.5000],
                [0.0000, 0.3333, 0.5000]])
    """
    if validate_args:
        _multiclass_stat_scores_arg_validation(num_classes, top_k, average, multidim_average, ignore_index)
        _multiclass_stat_scores_tensor_validation(preds, target, num_classes, multidim_average, ignore_index)
    preds, target = _multiclass_stat_scores_format(preds, target, top_k)
    tp, fp, tn, fn = _multiclass_stat_scores_update(
        preds, target, num_classes, top_k, average, multidim_average, ignore_index
    )
    return _accuracy_reduce(tp, fp, tn, fn, average=average, multidim_average=multidim_average)


def multilabel_accuracy(
    preds: Tensor,
    target: Tensor,
    num_labels: int,
    threshold: float = 0.5,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    r"""Computes `Accuracy`_ for multilabel tasks:

    .. math::
        \text{Accuracy} = \frac{1}{N}\sum_i^N 1(y_i = \hat{y}_i)

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a
    tensor of predictions.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, C, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Addtionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, C, ...)``

    The influence of the additional dimension ``...`` (if present) will be determined by the `multidim_average`
    argument.

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_labels: Integer specifing the number of labels
        threshold: Threshold for transforming probability to binary (0,1) predictions
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: Calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: Calculates statistic for each label and applies no reduction

        multidim_average:
            Defines how additionally dimensions ``...`` should be handled. Should be one of the following:

            - ``global``: Additional dimensions are flatted along the batch dimension
            - ``samplewise``: Statistic will be calculated independently for each sample on the ``N`` axis.
              The statistics in this case are calculated over the additional dimensions.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        The returned shape depends on the ``average`` and ``multidim_average`` arguments:

        - If ``multidim_average`` is set to ``global``:

          - If ``average='micro'/'macro'/'weighted'``, the output will be a scalar tensor
          - If ``average=None/'none'``, the shape will be ``(C,)``

        - If ``multidim_average`` is set to ``samplewise``:

          - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(N,)``
          - If ``average=None/'none'``, the shape will be ``(N, C)``

    Example (preds is int tensor):
        >>> from torchmetrics.functional.classification import multilabel_accuracy
        >>> target = torch.tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = torch.tensor([[0, 0, 1], [1, 0, 1]])
        >>> multilabel_accuracy(preds, target, num_labels=3)
        tensor(0.6667)
        >>> multilabel_accuracy(preds, target, num_labels=3, average=None)
        tensor([1.0000, 0.5000, 0.5000])

    Example (preds is float tensor):
        >>> from torchmetrics.functional.classification import multilabel_accuracy
        >>> target = torch.tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = torch.tensor([[0.11, 0.22, 0.84], [0.73, 0.33, 0.92]])
        >>> multilabel_accuracy(preds, target, num_labels=3)
        tensor(0.6667)
        >>> multilabel_accuracy(preds, target, num_labels=3, average=None)
        tensor([1.0000, 0.5000, 0.5000])

    Example (multidim tensors):
        >>> from torchmetrics.functional.classification import multilabel_accuracy
        >>> target = torch.tensor([[[0, 1], [1, 0], [0, 1]], [[1, 1], [0, 0], [1, 0]]])
        >>> preds = torch.tensor(
        ...     [
        ...         [[0.59, 0.91], [0.91, 0.99], [0.63, 0.04]],
        ...         [[0.38, 0.04], [0.86, 0.780], [0.45, 0.37]],
        ...     ]
        ... )
        >>> multilabel_accuracy(preds, target, num_labels=3, multidim_average='samplewise')
        tensor([0.3333, 0.1667])
        >>> multilabel_accuracy(preds, target, num_labels=3, multidim_average='samplewise', average=None)
        tensor([[0.5000, 0.5000, 0.0000],
                [0.0000, 0.0000, 0.5000]])
    """
    if validate_args:
        _multilabel_stat_scores_arg_validation(num_labels, threshold, average, multidim_average, ignore_index)
        _multilabel_stat_scores_tensor_validation(preds, target, num_labels, multidim_average, ignore_index)
    preds, target = _multilabel_stat_scores_format(preds, target, num_labels, threshold, ignore_index)
    tp, fp, tn, fn = _multilabel_stat_scores_update(preds, target, multidim_average)
    return _accuracy_reduce(tp, fp, tn, fn, average=average, multidim_average=multidim_average, multilabel=True)


def _check_subset_validity(mode: DataType) -> bool:
    """Checks input mode is valid."""
    return mode in (DataType.MULTILABEL, DataType.MULTIDIM_MULTICLASS)


def _mode(
    preds: Tensor,
    target: Tensor,
    threshold: float,
    top_k: Optional[int],
    num_classes: Optional[int],
    multiclass: Optional[bool],
    ignore_index: Optional[int] = None,
) -> DataType:
    """Finds the mode of the input tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        threshold: Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the
            case of binary or multi-label inputs. Default value of 0.5 corresponds to input being probabilities.
        top_k: Number of the highest probability or logit score predictions considered finding the correct label,
            relevant only for (multi-dimensional) multi-class inputs.
        num_classes: Number of classes. Necessary for ``'macro'``, ``'weighted'`` and ``None`` average methods.
        multiclass:
            Used only in certain special cases, where you want to treat inputs as a different type
            than what they appear to be.

    Example:
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> preds = torch.tensor([0, 2, 1, 3])
        >>> _mode(preds, target, 0.5, None, None, None)
        <DataType.MULTICLASS: 'multi-class'>
    """

    mode = _check_classification_inputs(
        preds,
        target,
        threshold=threshold,
        top_k=top_k,
        num_classes=num_classes,
        multiclass=multiclass,
        ignore_index=ignore_index,
    )
    return mode


def _accuracy_update(
    preds: Tensor,
    target: Tensor,
    reduce: Optional[str],
    mdmc_reduce: Optional[str],
    threshold: float,
    num_classes: Optional[int],
    top_k: Optional[int],
    multiclass: Optional[bool],
    ignore_index: Optional[int],
    mode: DataType,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Updates and returns stat scores (true positives, false positives, true negatives, false negatives) required
    to compute accuracy.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        reduce: Defines the reduction that is applied.
        mdmc_reduce: Defines how the multi-dimensional multi-class inputs are handled.
        threshold: Threshold for transforming probability or logit predictions to binary (0,1) predictions, in
            the case of binary or multi-label inputs.
        num_classes: Number of classes. Necessary for ``'macro'``, ``'weighted'`` and ``None`` average methods.
        top_k: Number of the highest probability or logit score predictions considered finding the correct label,
            relevant only for (multi-dimensional) multi-class inputs.
        multiclass: Used only in certain special cases, where you want to treat inputs as a different type
            than what they appear to be.
        ignore_index: Integer specifying a target class to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method. If an index is ignored, and ``average=None``
            or ``'none'``, the score for the ignored class will be returned as ``nan``.
        mode: Mode of the input tensors.
    """

    if mode == DataType.MULTILABEL and top_k:
        raise ValueError("You can not use the `top_k` parameter to calculate accuracy for multi-label inputs.")
    preds, target = _input_squeeze(preds, target)
    tp, fp, tn, fn = _stat_scores_update(
        preds,
        target,
        reduce=reduce,
        mdmc_reduce=mdmc_reduce,
        threshold=threshold,
        num_classes=num_classes,
        top_k=top_k,
        multiclass=multiclass,
        ignore_index=ignore_index,
        mode=mode,
    )
    return tp, fp, tn, fn


def _accuracy_compute(
    tp: Tensor,
    fp: Tensor,
    tn: Tensor,
    fn: Tensor,
    average: Optional[str],
    mdmc_average: Optional[str],
    mode: DataType,
) -> Tensor:
    """Computes accuracy from stat scores: true positives, false positives, true negatives, false negatives.

    Args:
        tp: True positives
        fp: False positives
        tn: True negatives
        fn: False negatives
        average: Defines the reduction that is applied.
        mdmc_average: Defines how averaging is done for multi-dimensional multi-class inputs (on top of the
            ``average`` parameter).
        mode: Mode of the input tensors

    Example:
        >>> preds = torch.tensor([0, 2, 1, 3])
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> threshold = 0.5
        >>> reduce = average = 'micro'
        >>> mdmc_average = 'global'
        >>> mode = _mode(preds, target, threshold, top_k=None, num_classes=None, multiclass=None)
        >>> tp, fp, tn, fn = _accuracy_update(
        ...                     preds,
        ...                     target,
        ...                     reduce,
        ...                     mdmc_average,
        ...                     threshold=0.5,
        ...                     num_classes=None,
        ...                     top_k=None,
        ...                     multiclass=None,
        ...                     ignore_index=None,
        ...                     mode=mode)
        >>> _accuracy_compute(tp, fp, tn, fn, average, mdmc_average, mode)
        tensor(0.5000)

        >>> target = torch.tensor([0, 1, 2])
        >>> preds = torch.tensor([[0.1, 0.9, 0], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3]])
        >>> top_k, threshold = 2, 0.5
        >>> reduce = average = 'micro'
        >>> mdmc_average = 'global'
        >>> mode = _mode(preds, target, threshold, top_k, num_classes=None, multiclass=None)
        >>> tp, fp, tn, fn = _accuracy_update(preds, target, reduce, mdmc_average, threshold,
        ...     num_classes=None, top_k=top_k, multiclass=None, ignore_index=None, mode=mode)
        >>> _accuracy_compute(tp, fp, tn, fn, average, mdmc_average, mode)
        tensor(0.6667)
    """

    simple_average = [AverageMethod.MICRO, AverageMethod.SAMPLES]
    if (mode == DataType.BINARY and average in simple_average) or mode == DataType.MULTILABEL:
        numerator = tp + tn
        denominator = tp + tn + fp + fn
    else:
        numerator = tp.clone()
        denominator = tp + fn

    if mdmc_average != MDMCAverageMethod.SAMPLEWISE:
        if average == AverageMethod.MACRO:
            cond = tp + fp + fn == 0
            numerator = numerator[~cond]
            denominator = denominator[~cond]

        if average == AverageMethod.NONE:
            # a class is not present if there exists no TPs, no FPs, and no FNs
            meaningless_indeces = torch.nonzero((tp | fn | fp) == 0).cpu()
            numerator[meaningless_indeces, ...] = -1
            denominator[meaningless_indeces, ...] = -1

    return _reduce_stat_scores(
        numerator=numerator,
        denominator=denominator,
        weights=None if average != AverageMethod.WEIGHTED else tp + fn,
        average=average,
        mdmc_average=mdmc_average,
    )


def _subset_accuracy_update(
    preds: Tensor,
    target: Tensor,
    threshold: float,
    top_k: Optional[int],
    ignore_index: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """Updates and returns variables required to compute subset accuracy.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        threshold: Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case
            of binary or multi-label inputs. Default value of ``0.5`` corresponds to input being probabilities.
        top_k: Number of the highest probability or logit score predictions considered finding the correct label,
            relevant only for (multi-dimensional) multi-class inputs.
    """

    preds, target = _input_squeeze(preds, target)
    preds, target, mode = _input_format_classification(
        preds, target, threshold=threshold, top_k=top_k, ignore_index=ignore_index
    )

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
    else:
        correct, total = tensor(0), tensor(0)

    return correct, total


def _subset_accuracy_compute(correct: Tensor, total: Tensor) -> Tensor:
    """Computes subset accuracy from number of correct observations and total number of observations.

    Args:
        correct: Number of correct observations
        total: Number of observations
    """

    return correct.float() / total


def accuracy(
    preds: Tensor,
    target: Tensor,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "micro",
    mdmc_average: Optional[str] = "global",
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    subset_accuracy: bool = False,
    num_classes: Optional[int] = None,
    multiclass: Optional[bool] = None,
    ignore_index: Optional[int] = None,
    task: Optional[Literal["binary", "multiclass", "multilabel"]] = None,
    num_labels: Optional[int] = None,
    multidim_average: Optional[Literal["global", "samplewise"]] = "global",
    validate_args: bool = True,
) -> Tensor:
    r"""Accuracy.

    .. note::
        From v0.10 an ``'binary_*'``, ``'multiclass_*'``, ``'multilabel_*'`` version now exist of each classification
        metric. Moving forward we recommend using these versions. This base metric will still work as it did
        prior to v0.10 until v0.11. From v0.11 the `task` argument introduced in this metric will be required
        and the general order of arguments may change, such that this metric will just function as an single
        entrypoint to calling the three specialized versions.

    Computes `Accuracy`_

    .. math::
        \text{Accuracy} = \frac{1}{N}\sum_i^N 1(y_i = \hat{y}_i)

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a
    tensor of predictions.

    For multi-class and multi-dimensional multi-class data with probability or logits predictions, the
    parameter ``top_k`` generalizes this metric to a Top-K accuracy metric: for each sample the
    top-K highest probability or logits items are considered to find the correct label.

    For multi-label and multi-dimensional multi-class inputs, this metric computes the "global"
    accuracy by default, which counts all labels or sub-samples separately. This can be
    changed to subset accuracy (which requires all labels or sub-samples in the sample to
    be correctly predicted) by setting ``subset_accuracy=True``.

    Accepts all input types listed in :ref:`pages/classification:input types`.

    Args:
        preds: Predictions from model (probabilities, logits or labels)
        target: Ground truth labels
        average:
            Defines the reduction that is applied. Should be one of the following:

            - ``'micro'`` [default]: Calculate the metric globally, across all samples and classes.
            - ``'macro'``: Calculate the metric for each class separately, and average the
              metrics across classes (with equal weights for each class).
            - ``'weighted'``: Calculate the metric for each class separately, and average the
              metrics across classes, weighting each class by its support (``tp + fn``).
            - ``'none'`` or ``None``: Calculate the metric for each class separately, and return
              the metric for every class.
            - ``'samples'``: Calculate the metric for each sample, and average the metrics
              across samples (with equal weights for each sample).

            .. note:: What is considered a sample in the multi-dimensional multi-class case
                depends on the value of ``mdmc_average``.

            .. note:: If ``'none'`` and a given class doesn't occur in the ``preds`` or ``target``,
                the value for the class will be ``nan``.

        mdmc_average:
            Defines how averaging is done for multi-dimensional multi-class inputs (on top of the
            ``average`` parameter). Should be one of the following:

            - ``None`` [default]: Should be left unchanged if your data is not multi-dimensional multi-class.

            - ``'samplewise'``: In this case, the statistics are computed separately for each
              sample on the ``N`` axis, and then averaged over samples.
              The computation for each sample is done by treating the flattened extra axes ``...``
              (see :ref:`pages/classification:input types`) as the ``N`` dimension within the sample,
              and computing the metric for the sample based on that.

            - ``'global'``: In this case the ``N`` and ``...`` dimensions of the inputs
              (see :ref:`pages/classification:input types`)
              are flattened into a new ``N_X`` sample axis, i.e. the inputs are treated as if they
              were ``(N_X, C)``. From here on the ``average`` parameter applies as usual.

        num_classes:
            Number of classes. Necessary for ``'macro'``, ``'weighted'`` and ``None`` average methods.

        threshold:
            Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case
            of binary or multi-label inputs. Default value of 0.5 corresponds to input being probabilities.
        top_k:
            Number of the highest probability or logit score predictions considered finding the correct label,
            relevant only for (multi-dimensional) multi-class inputs. The
            default value (``None``) will be interpreted as 1 for these inputs.

            Should be left at default (``None``) for all other types of inputs.
        multiclass:
            Used only in certain special cases, where you want to treat inputs as a different type
            than what they appear to be. See the parameter's
            :ref:`documentation section <pages/classification:using the multiclass parameter>`
            for a more detailed explanation and examples.
        ignore_index:
            Integer specifying a target class to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method. If an index is ignored, and ``average=None``
            or ``'none'``, the score for the ignored class will be returned as ``nan``.
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
        ValueError:
            If ``average`` is none of ``"micro"``, ``"macro"``, ``"weighted"``, ``"samples"``, ``"none"``, ``None``.
        ValueError:
            If ``mdmc_average`` is not one of ``None``, ``"samplewise"``, ``"global"``.
        ValueError:
            If ``average`` is set but ``num_classes`` is not provided.
        ValueError:
            If ``num_classes`` is set
            and ``ignore_index`` is not in the range ``[0, num_classes)``.
        ValueError:
            If ``top_k`` is not an ``integer`` larger than ``0``.

    Example:
        >>> import torch
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
    if task is not None:
        assert multidim_average is not None
        if task == "binary":
            return binary_accuracy(preds, target, threshold, multidim_average, ignore_index, validate_args)
        if task == "multiclass":
            assert isinstance(num_classes, int)
            assert isinstance(top_k, int)
            return multiclass_accuracy(
                preds, target, num_classes, average, top_k, multidim_average, ignore_index, validate_args
            )
        if task == "multilabel":
            assert isinstance(num_labels, int)
            return multilabel_accuracy(
                preds, target, num_labels, threshold, average, multidim_average, ignore_index, validate_args
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

    allowed_average = ["micro", "macro", "weighted", "samples", "none", None]
    if average not in allowed_average:
        raise ValueError(f"The `average` has to be one of {allowed_average}, got {average}.")

    if average in ["macro", "weighted", "none", None] and (not num_classes or num_classes < 1):
        raise ValueError(f"When you set `average` as {average}, you have to provide the number of classes.")

    allowed_mdmc_average = [None, "samplewise", "global"]
    if mdmc_average not in allowed_mdmc_average:
        raise ValueError(f"The `mdmc_average` has to be one of {allowed_mdmc_average}, got {mdmc_average}.")

    if num_classes and ignore_index is not None and (not ignore_index < num_classes or num_classes == 1):
        raise ValueError(f"The `ignore_index` {ignore_index} is not valid for inputs with {num_classes} classes")

    if top_k is not None and (not isinstance(top_k, int) or top_k <= 0):
        raise ValueError(f"The `top_k` should be an integer larger than 0, got {top_k}")

    preds, target = _input_squeeze(preds, target)
    mode = _mode(preds, target, threshold, top_k, num_classes, multiclass, ignore_index)
    reduce = "macro" if average in ["weighted", "none", None] else average

    if subset_accuracy and _check_subset_validity(mode):
        correct, total = _subset_accuracy_update(preds, target, threshold, top_k, ignore_index)
        return _subset_accuracy_compute(correct, total)
    tp, fp, tn, fn = _accuracy_update(
        preds, target, reduce, mdmc_average, threshold, num_classes, top_k, multiclass, ignore_index, mode
    )
    return _accuracy_compute(tp, fp, tn, fn, average, mdmc_average, mode)
