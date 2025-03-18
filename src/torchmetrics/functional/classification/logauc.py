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

from torchmetrics.functional.classification.roc import binary_roc, multiclass_roc, multilabel_roc
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.compute import _auc_compute_without_check, _safe_divide
from torchmetrics.utilities.data import interp
from torchmetrics.utilities.enums import ClassificationTask


def _validate_fpr_range(fpr_range: Tuple[float, float]) -> None:
    """Validate the `fpr_range` argument for the logauc metric."""
    if not isinstance(fpr_range, tuple) and not len(fpr_range) == 2:
        raise ValueError(f"The `fpr_range` should be a tuple of two floats, but got {type(fpr_range)}.")
    if not (0 <= fpr_range[0] < fpr_range[1] <= 1):
        raise ValueError(f"The `fpr_range` should be a tuple of two floats in the range [0, 1], but got {fpr_range}.")


def _binary_logauc_compute(
    fpr: Tensor,
    tpr: Tensor,
    fpr_range: Tuple[float, float] = (0.001, 0.1),
) -> Tensor:
    """Compute the logauc score for binary classification tasks."""
    fpr_range = torch.tensor(fpr_range).to(fpr.device)
    if fpr.numel() < 2 or tpr.numel() < 2:
        rank_zero_warn(
            "At least two values on for the fpr and tpr are required to compute the log AUC. Returns 0 score."
        )
        return torch.tensor(0.0, device=fpr.device)

    tpr = torch.cat([tpr, interp(fpr_range, fpr, tpr)]).sort().values
    fpr = torch.cat([fpr, fpr_range]).sort().values

    log_fpr = torch.log10(fpr)
    bounds = torch.log10(torch.tensor(fpr_range))

    lower_bound_idx = torch.where(log_fpr == bounds[0])[0][-1]
    upper_bound_idx = torch.where(log_fpr == bounds[1])[0][-1]

    trimmed_log_fpr = log_fpr[lower_bound_idx : upper_bound_idx + 1]
    trimmed_tpr = tpr[lower_bound_idx : upper_bound_idx + 1]

    # compute area and rescale it to the range of fpr
    return _auc_compute_without_check(trimmed_log_fpr, trimmed_tpr, 1.0) / (bounds[1] - bounds[0])


def _reduce_logauc(
    fpr: Union[Tensor, List[Tensor]],
    tpr: Union[Tensor, List[Tensor]],
    fpr_range: Tuple[float, float] = (0.001, 0.1),
    average: Optional[Literal["macro", "weighted", "none"]] = "macro",
    weights: Optional[Tensor] = None,
) -> Tensor:
    """Reduce the logauc score to a single value for multiclass and multilabel classification tasks."""
    scores = []
    for fpr_i, tpr_i in zip(fpr, tpr):
        scores.append(_binary_logauc_compute(fpr_i, tpr_i, fpr_range))
    scores = torch.stack(scores)
    if torch.isnan(scores).any():
        rank_zero_warn(
            "LogAUC score for one or more classes/labels was `nan`. Ignoring these classes in {average}-average."
        )
    idx = ~torch.isnan(scores)
    if average is None or average == "none":
        return scores
    if average == "macro":
        return scores[idx].mean()
    if average == "weighted" and weights is not None:
        weights = _safe_divide(weights[idx], weights[idx].sum())
        return (scores[idx] * weights).sum()
    raise ValueError(f"Got unknown average parameter: {average}. Please choose one of ['macro', 'weighted', 'none'].")


def binary_logauc(
    preds: Tensor,
    target: Tensor,
    fpr_range: Tuple[float, float] = (0.001, 0.1),
    thresholds: Optional[Union[int, List[float], Tensor]] = None,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    r"""Compute the `Log AUC`_ score for binary classification tasks.

    The score is computed by first computing the ROC curve, which then is interpolated to the specified range of false
    positive rates (FPR) and then the log is taken of the FPR before the area under the curve (AUC) is computed. The
    score is commonly used in applications where the positive and negative are imbalanced and a low false positive rate
    is of high importance.

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

    Args:
        preds: Tensor with predictions
        target: Tensor with ground truth labels
        fpr_range: 2-element tuple with the lower and upper bound of the false positive rate range to compute the log
            AUC score.
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
        A single scalar with the log auc score

    Example:
        >>> from torchmetrics.functional.classification import binary_logauc
        >>> from torch import tensor
        >>> preds = tensor([0.75, 0.05, 0.05, 0.05, 0.05])
        >>> target = tensor([1, 0, 0, 0, 0])
        >>> binary_logauc(preds, target)
        tensor(1.)

    """
    _validate_fpr_range(fpr_range)
    fpr, tpr, _ = binary_roc(preds, target, thresholds, ignore_index, validate_args)
    return _binary_logauc_compute(fpr, tpr, fpr_range)


def multiclass_logauc(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    fpr_range: Tuple[float, float] = (0.001, 0.1),
    average: Optional[Literal["macro", "none"]] = "macro",
    thresholds: Optional[Union[int, List[float], Tensor]] = None,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    r"""Compute the `Log AUC`_ score for multiclass classification tasks.

    The score is computed by first computing the ROC curve, which then is interpolated to the specified range of false
    positive rates (FPR) and then the log is taken of the FPR before the area under the curve (AUC) is computed. The
    score is commonly used in applications where the positive and negative are imbalanced and a low false positive rate
    is of high importance.

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
        fpr_range: 2-element tuple with the lower and upper bound of the false positive rate range to compute the log
            AUC score.
        thresholds:
            Can be one of:

            - If set to `None`, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an `int` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an `list` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d `tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

        average:
            Defines the reduction that is applied over classes. Should be one of the following:

            - ``macro``: Calculate score for each class and average them
            - ``"none"`` or ``None``: calculates score for each class and applies no reduction

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Example:
        >>> from torchmetrics.functional.classification import multiclass_logauc
        >>> preds = torch.tensor([[0.75, 0.05, 0.05, 0.05, 0.05],
        ...                       [0.05, 0.75, 0.05, 0.05, 0.05],
        ...                       [0.05, 0.05, 0.75, 0.05, 0.05],
        ...                       [0.05, 0.05, 0.05, 0.75, 0.05]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> multiclass_logauc(preds, target, num_classes=5, average="macro", thresholds=None)
        tensor(0.4000)
        >>> multiclass_logauc(preds, target, num_classes=5, average=None, thresholds=None)
        tensor([1., 1., 0., 0., 0.])

    """
    if validate_args:
        _validate_fpr_range(fpr_range)
    fpr, tpr, _ = multiclass_roc(
        preds, target, num_classes, thresholds, average=None, ignore_index=ignore_index, validate_args=validate_args
    )
    return _reduce_logauc(fpr, tpr, fpr_range, average)


def multilabel_logauc(
    preds: Tensor,
    target: Tensor,
    num_labels: int,
    fpr_range: Tuple[float, float] = (0.001, 0.1),
    average: Optional[Literal["macro", "none"]] = "macro",
    thresholds: Optional[Union[int, List[float], Tensor]] = None,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    r"""Compute the `Log AUC`_ score for multilabel classification tasks.

    The score is computed by first computing the ROC curve, which then is interpolated to the specified range of false
    positive rates (FPR) and then the log is taken of the FPR before the area under the curve (AUC) is computed. The
    score is commonly used in applications where the positive and negative are imbalanced and a low false positive rate
    is of high importance.

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
        fpr_range: 2-element tuple with the lower and upper bound of the false positive rate range to compute the log
            AUC score.
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``macro``: Calculate score for each label and average them
            - ``"none"`` or ``None``: calculates score for each label and applies no reduction

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

    Example:
        >>> from torchmetrics.functional.classification import multilabel_logauc
        >>> preds = torch.tensor([[0.75, 0.05, 0.35],
        ...                       [0.45, 0.75, 0.05],
        ...                       [0.05, 0.55, 0.75],
        ...                       [0.05, 0.65, 0.05]])
        >>> target = torch.tensor([[1, 0, 1],
        ...                        [0, 0, 0],
        ...                        [0, 1, 1],
        ...                        [1, 1, 1]])
        >>> multilabel_logauc(preds, target, num_labels=3, average="macro", thresholds=None)
        tensor(0.3945)
        >>> multilabel_logauc(preds, target, num_labels=3, average=None, thresholds=None)
        tensor([0.5000, 0.0000, 0.6835])

    """
    fpr, tpr, _ = multilabel_roc(preds, target, num_labels, thresholds, ignore_index, validate_args)
    return _reduce_logauc(fpr, tpr, fpr_range, average=average)


def logauc(
    preds: Tensor,
    target: Tensor,
    task: Literal["binary", "multiclass", "multilabel"],
    thresholds: Optional[Union[int, List[float], Tensor]] = None,
    num_classes: Optional[int] = None,
    num_labels: Optional[int] = None,
    fpr_range: Tuple[float, float] = (0.001, 0.1),
    average: Optional[Literal["macro", "none"]] = None,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Optional[Tensor]:
    r"""Compute the `Log AUC`_ score for classification tasks.

    The score is computed by first computing the ROC curve, which then is interpolated to the specified range of false
    positive rates (FPR) and then the log is taken of the FPR before the area under the curve (AUC) is computed. The
    score is commonly used in applications where the positive and negative are imbalanced and a low false positive rate
    is of high importance.

    """
    task = ClassificationTask.from_str(task)
    if task == ClassificationTask.BINARY:
        return binary_logauc(preds, target, fpr_range, thresholds, ignore_index, validate_args)
    if task == ClassificationTask.MULTICLASS:
        if not isinstance(num_classes, int):
            raise ValueError(f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`")
        return multiclass_logauc(
            preds, target, num_classes, fpr_range, average, thresholds, ignore_index, validate_args
        )
    if task == ClassificationTask.MULTILABEL:
        if not isinstance(num_labels, int):
            raise ValueError(f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`")
        return multilabel_logauc(preds, target, num_labels, fpr_range, average, thresholds, ignore_index, validate_args)
    return None
