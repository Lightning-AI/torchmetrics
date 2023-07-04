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
from typing import Optional, Tuple

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.classification.stat_scores import (
    _multiclass_stat_scores_arg_validation,
    _multiclass_stat_scores_format,
    _multiclass_stat_scores_tensor_validation,
    _multilabel_stat_scores_arg_validation,
    _multilabel_stat_scores_format,
    _multilabel_stat_scores_tensor_validation,
)
from torchmetrics.utilities.compute import _safe_divide
from torchmetrics.utilities.enums import ClassificationTaskNoBinary


def _exact_match_reduce(
    correct: Tensor,
    total: Tensor,
) -> Tensor:
    """Reduce exact match."""
    return _safe_divide(correct, total)


def _multiclass_exact_match_update(
    preds: Tensor,
    target: Tensor,
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """Compute the statistics."""
    if ignore_index is not None:
        preds = preds.clone()
        preds[target == ignore_index] = ignore_index

    correct = (preds == target).sum(1) == preds.shape[1]
    correct = correct if multidim_average == "samplewise" else correct.sum()
    total = torch.tensor(preds.shape[0] if multidim_average == "global" else 1, device=correct.device)
    return correct, total


def multiclass_exact_match(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    r"""Compute Exact match (also known as subset accuracy) for multiclass tasks.

    Exact Match is a stricter version of accuracy where all labels have to match exactly for the sample to be
    correctly classified.

    Accepts the following input tensors:

    - ``preds``: ``(N, ...)`` (int tensor) or ``(N, C, ..)`` (float tensor). If preds is a floating point
      we apply ``torch.argmax`` along the ``C`` dimension to automatically convert probabilities/logits into
      an int tensor.
    - ``target`` (int tensor): ``(N, ...)``

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_classes: Integer specifing the number of labels
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
        The returned shape depends on the ``multidim_average`` argument:

        - If ``multidim_average`` is set to ``global`` the output will be a scalar tensor
        - If ``multidim_average`` is set to ``samplewise`` the output will be a tensor of shape ``(N,)``

    Example (multidim tensors):
        >>> from torch import tensor
        >>> from torchmetrics.functional.classification import multiclass_exact_match
        >>> target = tensor([[[0, 1], [2, 1], [0, 2]], [[1, 1], [2, 0], [1, 2]]])
        >>> preds = tensor([[[0, 1], [2, 1], [0, 2]], [[2, 2], [2, 1], [1, 0]]])
        >>> multiclass_exact_match(preds, target, num_classes=3, multidim_average='global')
        tensor(0.5000)

    Example (multidim tensors):
        >>> from torchmetrics.functional.classification import multiclass_exact_match
        >>> target = tensor([[[0, 1], [2, 1], [0, 2]], [[1, 1], [2, 0], [1, 2]]])
        >>> preds = tensor([[[0, 1], [2, 1], [0, 2]], [[2, 2], [2, 1], [1, 0]]])
        >>> multiclass_exact_match(preds, target, num_classes=3, multidim_average='samplewise')
        tensor([1., 0.])
    """
    top_k, average = 1, None
    if validate_args:
        _multiclass_stat_scores_arg_validation(num_classes, top_k, average, multidim_average, ignore_index)
        _multiclass_stat_scores_tensor_validation(preds, target, num_classes, multidim_average, ignore_index)
    preds, target = _multiclass_stat_scores_format(preds, target, top_k)
    correct, total = _multiclass_exact_match_update(preds, target, multidim_average, ignore_index)
    return _exact_match_reduce(correct, total)


def _multilabel_exact_match_update(
    preds: Tensor, target: Tensor, num_labels: int, multidim_average: Literal["global", "samplewise"] = "global"
) -> Tuple[Tensor, Tensor]:
    """Compute the statistics."""
    if multidim_average == "global":
        preds = torch.movedim(preds, 1, -1).reshape(-1, num_labels)
        target = torch.movedim(target, 1, -1).reshape(-1, num_labels)

    correct = ((preds == target).sum(1) == num_labels).sum(dim=-1)
    total = torch.tensor(preds.shape[0 if multidim_average == "global" else 2], device=correct.device)
    return correct, total


def multilabel_exact_match(
    preds: Tensor,
    target: Tensor,
    num_labels: int,
    threshold: float = 0.5,
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    r"""Compute Exact match (also known as subset accuracy) for multilabel tasks.

    Exact Match is a stricter version of accuracy where all labels have to match exactly for the sample to be
    correctly classified.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, C, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Addtionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, C, ...)``

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_labels: Integer specifing the number of labels
        threshold: Threshold for transforming probability to binary (0,1) predictions
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
        The returned shape depends on the ``multidim_average`` argument:

        - If ``multidim_average`` is set to ``global`` the output will be a scalar tensor
        - If ``multidim_average`` is set to ``samplewise`` the output will be a tensor of shape ``(N,)``

    Example (preds is int tensor):
        >>> from torch import tensor
        >>> from torchmetrics.functional.classification import multilabel_exact_match
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0, 0, 1], [1, 0, 1]])
        >>> multilabel_exact_match(preds, target, num_labels=3)
        tensor(0.5000)

    Example (preds is float tensor):
        >>> from torchmetrics.functional.classification import multilabel_exact_match
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0.11, 0.22, 0.84], [0.73, 0.33, 0.92]])
        >>> multilabel_exact_match(preds, target, num_labels=3)
        tensor(0.5000)

    Example (multidim tensors):
        >>> from torchmetrics.functional.classification import multilabel_exact_match
        >>> target = tensor([[[0, 1], [1, 0], [0, 1]], [[1, 1], [0, 0], [1, 0]]])
        >>> preds = tensor([[[0.59, 0.91], [0.91, 0.99], [0.63, 0.04]],
        ...                 [[0.38, 0.04], [0.86, 0.780], [0.45, 0.37]]])
        >>> multilabel_exact_match(preds, target, num_labels=3, multidim_average='samplewise')
        tensor([0., 0.])
    """
    average = None
    if validate_args:
        _multilabel_stat_scores_arg_validation(num_labels, threshold, average, multidim_average, ignore_index)
        _multilabel_stat_scores_tensor_validation(preds, target, num_labels, multidim_average, ignore_index)
    preds, target = _multilabel_stat_scores_format(preds, target, num_labels, threshold, ignore_index)
    correct, total = _multilabel_exact_match_update(preds, target, num_labels, multidim_average)
    return _exact_match_reduce(correct, total)


def exact_match(
    preds: Tensor,
    target: Tensor,
    task: Literal["multiclass", "multilabel"],
    num_classes: Optional[int] = None,
    num_labels: Optional[int] = None,
    threshold: float = 0.5,
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    r"""Compute Exact match (also known as subset accuracy).

    Exact Match is a stricter version of accuracy where all classes/labels have to match exactly for the sample to be
    correctly classified.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'multiclass'`` or ``'multilabel'``. See the documentation of
    :func:`multiclass_exact_match` and :func:`multilabel_exact_match` for the specific details of
    each argument influence and examples.
    Legacy Example:
        >>> from torch import tensor
        >>> target = tensor([[[0, 1], [2, 1], [0, 2]], [[1, 1], [2, 0], [1, 2]]])
        >>> preds = tensor([[[0, 1], [2, 1], [0, 2]], [[2, 2], [2, 1], [1, 0]]])
        >>> exact_match(preds, target, task="multiclass", num_classes=3, multidim_average='global')
        tensor(0.5000)

        >>> target = tensor([[[0, 1], [2, 1], [0, 2]], [[1, 1], [2, 0], [1, 2]]])
        >>> preds = tensor([[[0, 1], [2, 1], [0, 2]], [[2, 2], [2, 1], [1, 0]]])
        >>> exact_match(preds, target, task="multiclass", num_classes=3, multidim_average='samplewise')
        tensor([1., 0.])
    """
    task = ClassificationTaskNoBinary.from_str(task)
    if task == ClassificationTaskNoBinary.MULTICLASS:
        assert num_classes is not None  # noqa: S101  # needed for mypy
        return multiclass_exact_match(preds, target, num_classes, multidim_average, ignore_index, validate_args)
    if task == ClassificationTaskNoBinary.MULTILABEL:
        assert num_labels is not None  # noqa: S101  # needed for mypy
        return multilabel_exact_match(
            preds, target, num_labels, threshold, multidim_average, ignore_index, validate_args
        )
    raise ValueError(f"Not handled value: {task}")
