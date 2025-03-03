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

from torch import Tensor
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
)
from torchmetrics.utilities.compute import _adjust_weights_safe_divide, _safe_divide
from torchmetrics.utilities.enums import ClassificationTask


def _accuracy_reduce(
    tp: Tensor,
    fp: Tensor,
    tn: Tensor,
    fn: Tensor,
    average: Optional[Literal["binary", "micro", "macro", "weighted", "none"]],
    multidim_average: Literal["global", "samplewise"] = "global",
    multilabel: bool = False,
    top_k: int = 1,
) -> Tensor:
    """Reduce classification statistics into accuracy score.

    Args:
        tp: number of true positives
        fp: number of false positives
        tn: number of true negatives
        fn: number of false negatives
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``binary``: for binary reduction
            - ``micro``: sum score over all classes/labels
            - ``macro``: salculate score for each class/label and average them
            - ``weighted``: calculates score for each class/label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates score for each class/label and applies no reduction

        multidim_average:
            Defines how additionally dimensions ``...`` should be handled. Should be one of the following:

            - ``global``: Additional dimensions are flatted along the batch dimension
            - ``samplewise``: Statistic will be calculated independently for each sample on the ``N`` axis.

        multilabel: If input is multilabel or not
        top_k: value for top-k accuracy, else 1

    Returns:
        Accuracy score

    """
    if average == "binary":
        return _safe_divide(tp + tn, tp + tn + fp + fn)
    if average == "micro":
        tp = tp.sum(dim=0 if multidim_average == "global" else 1)
        fn = fn.sum(dim=0 if multidim_average == "global" else 1)
        if multilabel:
            fp = fp.sum(dim=0 if multidim_average == "global" else 1)
            tn = tn.sum(dim=0 if multidim_average == "global" else 1)
            return _safe_divide(tp + tn, tp + tn + fp + fn)
        return _safe_divide(tp, tp + fn)

    score = _safe_divide(tp + tn, tp + tn + fp + fn) if multilabel else _safe_divide(tp, tp + fn)
    return _adjust_weights_safe_divide(score, average, multilabel, tp, fp, fn, top_k)


def binary_accuracy(
    preds: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    r"""Compute `Accuracy`_ for binary tasks.

    .. math::
        \text{Accuracy} = \frac{1}{N}\sum_i^N 1(y_i = \hat{y}_i)

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a
    tensor of predictions.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Additionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, ...)``

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
        >>> from torch import tensor
        >>> from torchmetrics.functional.classification import binary_accuracy
        >>> target = tensor([0, 1, 0, 1, 0, 1])
        >>> preds = tensor([0, 0, 1, 1, 0, 1])
        >>> binary_accuracy(preds, target)
        tensor(0.6667)

    Example (preds is float tensor):
        >>> from torchmetrics.functional.classification import binary_accuracy
        >>> target = tensor([0, 1, 0, 1, 0, 1])
        >>> preds = tensor([0.11, 0.22, 0.84, 0.73, 0.33, 0.92])
        >>> binary_accuracy(preds, target)
        tensor(0.6667)

    Example (multidim tensors):
        >>> from torchmetrics.functional.classification import binary_accuracy
        >>> target = tensor([[[0, 1], [1, 0], [0, 1]], [[1, 1], [0, 0], [1, 0]]])
        >>> preds = tensor([[[0.59, 0.91], [0.91, 0.99], [0.63, 0.04]],
        ...                 [[0.38, 0.04], [0.86, 0.780], [0.45, 0.37]]])
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
    num_classes: Optional[int] = None,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
    top_k: int = 1,
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    r"""Compute `Accuracy`_ for multiclass tasks.

    .. math::
        \text{Accuracy} = \frac{1}{N}\sum_i^N 1(y_i = \hat{y}_i)

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a
    tensor of predictions.

    Accepts the following input tensors:

    - ``preds``: ``(N, ...)`` (int tensor) or ``(N, C, ..)`` (float tensor). If preds is a floating point
      we apply ``torch.argmax`` along the ``C`` dimension to automatically convert probabilities/logits into
      an int tensor.
    - ``target`` (int tensor): ``(N, ...)``

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_classes: Integer specifying the number of classes
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates statistic for each label and applies no reduction

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
        >>> from torch import tensor
        >>> from torchmetrics.functional.classification import multiclass_accuracy
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([2, 1, 0, 1])
        >>> multiclass_accuracy(preds, target, num_classes=3)
        tensor(0.8333)
        >>> multiclass_accuracy(preds, target, num_classes=3, average=None)
        tensor([0.5000, 1.0000, 1.0000])

    Example (preds is float tensor):
        >>> from torchmetrics.functional.classification import multiclass_accuracy
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([[0.16, 0.26, 0.58],
        ...                 [0.22, 0.61, 0.17],
        ...                 [0.71, 0.09, 0.20],
        ...                 [0.05, 0.82, 0.13]])
        >>> multiclass_accuracy(preds, target, num_classes=3)
        tensor(0.8333)
        >>> multiclass_accuracy(preds, target, num_classes=3, average=None)
        tensor([0.5000, 1.0000, 1.0000])

    Example (multidim tensors):
        >>> from torchmetrics.functional.classification import multiclass_accuracy
        >>> target = tensor([[[0, 1], [2, 1], [0, 2]], [[1, 1], [2, 0], [1, 2]]])
        >>> preds = tensor([[[0, 2], [2, 0], [0, 1]], [[2, 2], [2, 1], [1, 0]]])
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
        preds, target, num_classes or 1, top_k, average, multidim_average, ignore_index
    )
    return _accuracy_reduce(tp, fp, tn, fn, average=average, multidim_average=multidim_average, top_k=top_k)


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
    r"""Compute `Accuracy`_ for multilabel tasks.

    .. math::
        \text{Accuracy} = \frac{1}{N}\sum_i^N 1(y_i = \hat{y}_i)

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a
    tensor of predictions.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, C, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Additionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, C, ...)``

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_labels: Integer specifying the number of labels
        threshold: Threshold for transforming probability to binary (0,1) predictions
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates statistic for each label and applies no reduction

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
        >>> from torch import tensor
        >>> from torchmetrics.functional.classification import multilabel_accuracy
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0, 0, 1], [1, 0, 1]])
        >>> multilabel_accuracy(preds, target, num_labels=3)
        tensor(0.6667)
        >>> multilabel_accuracy(preds, target, num_labels=3, average=None)
        tensor([1.0000, 0.5000, 0.5000])

    Example (preds is float tensor):
        >>> from torchmetrics.functional.classification import multilabel_accuracy
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0.11, 0.22, 0.84], [0.73, 0.33, 0.92]])
        >>> multilabel_accuracy(preds, target, num_labels=3)
        tensor(0.6667)
        >>> multilabel_accuracy(preds, target, num_labels=3, average=None)
        tensor([1.0000, 0.5000, 0.5000])

    Example (multidim tensors):
        >>> from torchmetrics.functional.classification import multilabel_accuracy
        >>> target = tensor([[[0, 1], [1, 0], [0, 1]], [[1, 1], [0, 0], [1, 0]]])
        >>> preds = tensor([[[0.59, 0.91], [0.91, 0.99], [0.63, 0.04]],
        ...                 [[0.38, 0.04], [0.86, 0.780], [0.45, 0.37]]])
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


def accuracy(
    preds: Tensor,
    target: Tensor,
    task: Literal["binary", "multiclass", "multilabel"],
    threshold: float = 0.5,
    num_classes: Optional[int] = None,
    num_labels: Optional[int] = None,
    average: Literal["micro", "macro", "weighted", "none"] = "micro",
    multidim_average: Literal["global", "samplewise"] = "global",
    top_k: Optional[int] = 1,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    r"""Compute `Accuracy`_.

    .. math::
        \text{Accuracy} = \frac{1}{N}\sum_i^N 1(y_i = \hat{y}_i)

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a tensor of predictions.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``multilabel``. See the documentation of
    :func:`~torchmetrics.functional.classification.binary_accuracy`,
    :func:`~torchmetrics.functional.classification.multiclass_accuracy` and
    :func:`~torchmetrics.functional.classification.multilabel_accuracy` for the specific details of
    each argument influence and examples.

    Legacy Example:
        >>> from torch import tensor
        >>> target = tensor([0, 1, 2, 3])
        >>> preds = tensor([0, 2, 1, 3])
        >>> accuracy(preds, target, task="multiclass", num_classes=4)
        tensor(0.5000)

        >>> target = tensor([0, 1, 2])
        >>> preds = tensor([[0.1, 0.9, 0], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3]])
        >>> accuracy(preds, target, task="multiclass", num_classes=3, top_k=2)
        tensor(0.6667)

    """
    task = ClassificationTask.from_str(task)

    if task == ClassificationTask.BINARY:
        return binary_accuracy(preds, target, threshold, multidim_average, ignore_index, validate_args)
    if task == ClassificationTask.MULTICLASS:
        if not isinstance(num_classes, int):
            raise ValueError(
                f"Optional arg `num_classes` must be type `int` when task is {task}. Got {type(num_classes)}"
            )
        if not isinstance(top_k, int):
            raise ValueError(f"Optional arg `top_k` must be type `int` when task is {task}. Got {type(top_k)}")
        return multiclass_accuracy(
            preds, target, num_classes, average, top_k, multidim_average, ignore_index, validate_args
        )
    if task == ClassificationTask.MULTILABEL:
        if not isinstance(num_labels, int):
            raise ValueError(
                f"Optional arg `num_labels` must be type `int` when task is {task}. Got {type(num_labels)}"
            )
        return multilabel_accuracy(
            preds, target, num_labels, threshold, average, multidim_average, ignore_index, validate_args
        )
    raise ValueError(f"Not handled value: {task}")
