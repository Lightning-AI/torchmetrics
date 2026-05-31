# Copyright The Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict, Optional

import torch
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
from torchmetrics.utilities.compute import _safe_divide


def _compute_per_class_metrics(
    tp: Tensor, fp: Tensor, tn: Tensor, fn: Tensor, zero_division: float = 0.0
) -> Dict[str, Tensor]:
    """Compute precision, recall, f1-score, support from stat scores.

    Args:
        tp: True positives, shape ``(C,)`` or scalar
        fp: False positives, shape ``(C,)`` or scalar
        tn: True negatives, shape ``(C,)`` or scalar
        fn: False negatives, shape ``(C,)`` or scalar
        zero_division: Value to return when division by zero (0 or 1)

    Returns:
        Dictionary with keys ``precision``, ``recall``, ``f1_score``, ``support``
    """
    support = tp + fn
    precision = _safe_divide(tp.float(), tp.float() + fp.float(), zero_division)
    recall = _safe_divide(tp.float(), tp.float() + fn.float(), zero_division)
    f1_score = _safe_divide(
        2 * tp.float(), 2 * tp.float() + fp.float() + fn.float(), zero_division
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "support": support,
    }


def _compute_average(
    scores: Dict[str, Tensor],
    average: Literal["micro", "macro", "weighted"],
    tp: Tensor,
    fp: Tensor,
    fn: Tensor,
) -> Dict[str, Tensor]:
    """Compute averaged metrics from per-class scores.

    Args:
        scores: Per-class metric dict from ``_compute_per_class_metrics``
        average: Averaging method: ``micro``, ``macro``, or ``weighted``
        tp: True positives per class
        fp: False positives per class
        fn: False negatives per class

    Returns:
        Dictionary of scalar tensors for each metric
    """
    result: Dict[str, Tensor] = {}
    support = scores["support"]

    if average == "micro":
        tp_sum = tp.sum()
        fp_sum = fp.sum()
        fn_sum = fn.sum()
        result["precision"] = _safe_divide(tp_sum.float(), tp_sum.float() + fp_sum.float(), 0.0)
        result["recall"] = _safe_divide(tp_sum.float(), tp_sum.float() + fn_sum.float(), 0.0)
        result["f1_score"] = _safe_divide(
            2 * tp_sum.float(), 2 * tp_sum.float() + fp_sum.float() + fn_sum.float(), 0.0
        )
        result["support"] = support.sum()
    elif average == "macro":
        for key in ("precision", "recall", "f1_score"):
            result[key] = scores[key].float().mean()
        result["support"] = support.sum()
    elif average == "weighted":
        weights = support.float()
        weights_sum = weights.sum()
        for key in ("precision", "recall", "f1_score"):
            if weights_sum > 0:
                result[key] = _safe_divide((weights * scores[key]).sum(), weights_sum, 0.0)
            else:
                result[key] = torch.tensor(0.0)
        result["support"] = support.sum()

    return result


def binary_classification_report(
    preds: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
    zero_division: float = 0.0,
) -> Dict[str, Dict[str, Tensor]]:
    r"""Compute a classification report for binary tasks.

    Generates per-class and average metrics (precision, recall, F1-score, support)
    similar to ``sklearn.metrics.classification_report`` but using PyTorch tensors.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): An int or float tensor of shape ``(N, ...)``.
      If preds is a floating point tensor with values outside [0,1] range we consider
      the input to be logits and will auto apply sigmoid per element. Additionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, ...)``

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        threshold: Threshold for transforming probability to binary {0,1} predictions
        multidim_average:
            Defines how additionally dimensions ``...`` should be handled. Should be one of:

            - ``global``: Additional dimensions are flatted along the batch dimension
            - ``samplewise``: Statistic will be calculated independently for each sample

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        zero_division: Value to return when there is a zero division. Should be 0 or 1.

    Returns:
        A dictionary with keys ``"0"``, ``"1"``, ``"macro"``, ``"weighted"``.
        Each value is a dict with keys ``"precision"``, ``"recall"``, ``"f1_score"``, ``"support"``.

    Example (preds is int tensor):
        >>> from torch import tensor
        >>> from torchmetrics.functional.classification import binary_classification_report
        >>> target = tensor([0, 1, 0, 1, 0, 1])
        >>> preds = tensor([0, 0, 1, 1, 0, 1])
        >>> report = binary_classification_report(preds, target)
        >>> report["0"]["precision"]
        tensor(0.6667)
        >>> report["1"]["precision"]
        tensor(0.6667)
        >>> report["macro"]["f1_score"]
        tensor(0.6667)

    Example (preds is float tensor):
        >>> from torch import tensor
        >>> from torchmetrics.functional.classification import binary_classification_report
        >>> target = tensor([0, 1, 0, 1, 0, 1])
        >>> preds = tensor([0.11, 0.22, 0.84, 0.73, 0.33, 0.92])
        >>> report = binary_classification_report(preds, target)
        >>> report["0"]["precision"]
        tensor(0.6667)
        >>> report["1"]["precision"]
        tensor(0.6667)

    """
    if validate_args:
        _binary_stat_scores_arg_validation(threshold, multidim_average, ignore_index, zero_division)
        _binary_stat_scores_tensor_validation(preds, target, multidim_average, ignore_index)
    preds, target = _binary_stat_scores_format(preds, target, threshold, ignore_index)
    tp, fp, tn, fn = _binary_stat_scores_update(preds, target, multidim_average)

    # For binary, compute metrics for both classes (0 and 1)
    # Class 0 perspective: positive = label 0 → TP=TN_orig, FP=FN_orig, FN=FP_orig
    tp0, fp0, fn0 = tn, fn, fp  # class 0: positive = label 0
    tp1, fp1, fn1 = tp, fp, fn  # class 1: positive = label 1

    # Stack per-class stats: shape (2,) each
    tp_per_class = torch.stack([tp0, tp1])
    fp_per_class = torch.stack([fp0, fp1])
    fn_per_class = torch.stack([fn0, fn1])
    tn_per_class = torch.stack([tn, tp])

    per_class = _compute_per_class_metrics(tp_per_class, fp_per_class, tn_per_class, fn_per_class, zero_division)
    macro_avg = _compute_average(per_class, "macro", tp_per_class, fp_per_class, fn_per_class)
    weighted_avg = _compute_average(per_class, "weighted", tp_per_class, fp_per_class, fn_per_class)

    def _extract(d: Dict[str, Tensor], idx: int) -> Dict[str, Tensor]:
        return {k: v[idx] if v.ndim > 0 else v for k, v in d.items()}

    return {
        "0": _extract(per_class, 0),
        "1": _extract(per_class, 1),
        "macro": dict(macro_avg),
        "weighted": dict(weighted_avg),
    }


def multiclass_classification_report(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    top_k: int = 1,
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
    zero_division: float = 0.0,
) -> Dict[str, Dict[str, Tensor]]:
    r"""Compute a classification report for multiclass tasks.

    Generates per-class and average metrics (precision, recall, F1-score, support)
    similar to ``sklearn.metrics.classification_report`` but using PyTorch tensors.

    Accepts the following input tensors:

    - ``preds``: ``(N, ...)`` (int tensor) or ``(N, C, ..)`` (float tensor). If preds
      is a floating point we apply ``torch.argmax`` along the ``C`` dimension to
      automatically convert probabilities/logits into an int tensor.
    - ``target`` (int tensor): ``(N, ...)``

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_classes: Integer specifying the number of classes
        top_k:
            Number of highest probability or logit score predictions considered to find
            the correct label. Only works when ``preds`` contain probabilities/logits.
        multidim_average:
            Defines how additionally dimensions ``...`` should be handled. Should be one of:

            - ``global``: Additional dimensions are flatted along the batch dimension
            - ``samplewise``: Statistic will be calculated independently for each sample

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        zero_division: Value to return when there is a zero division. Should be 0 or 1.

    Returns:
        A dictionary with per-class keys ``"0"``, ``"1"``, ..., ``"{C-1}"`` and
        summary keys ``"micro"``, ``"macro"``, ``"weighted"``.
        Each value is a dict with keys ``"precision"``, ``"recall"``, ``"f1_score"``, ``"support"``.

    Example (preds is int tensor):
        >>> from torch import tensor
        >>> from torchmetrics.functional.classification import multiclass_classification_report
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([2, 1, 0, 1])
        >>> report = multiclass_classification_report(preds, target, num_classes=3)
        >>> report["0"]["precision"]
        tensor(1.)
        >>> report["1"]["recall"]
        tensor(1.)

    Example (preds is float tensor):
        >>> from torch import tensor
        >>> from torchmetrics.functional.classification import multiclass_classification_report
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([[0.16, 0.26, 0.58],
        ...                 [0.22, 0.61, 0.17],
        ...                 [0.71, 0.09, 0.20],
        ...                 [0.05, 0.82, 0.13]])
        >>> report = multiclass_classification_report(preds, target, num_classes=3)
        >>> report["0"]["precision"]
        tensor(1.)

    """
    # Use average=None to get per-class stats
    _average = None

    if validate_args:
        _multiclass_stat_scores_arg_validation(num_classes, top_k, _average, multidim_average, ignore_index)
        _multiclass_stat_scores_tensor_validation(preds, target, num_classes, multidim_average, ignore_index)
    preds, target = _multiclass_stat_scores_format(preds, target, top_k)
    tp, fp, tn, fn = _multiclass_stat_scores_update(
        preds, target, num_classes, top_k, _average, multidim_average, ignore_index
    )

    per_class = _compute_per_class_metrics(tp, fp, tn, fn, zero_division)
    result: Dict[str, Dict[str, Tensor]] = {}

    # Per-class entries
    for c in range(num_classes):
        result[str(c)] = {
            "precision": per_class["precision"][c],
            "recall": per_class["recall"][c],
            "f1_score": per_class["f1_score"][c],
            "support": per_class["support"][c],
        }

    # Summary averages (always include all)
    result["micro"] = _compute_average(per_class, "micro", tp, fp, fn)
    result["macro"] = _compute_average(per_class, "macro", tp, fp, fn)
    result["weighted"] = _compute_average(per_class, "weighted", tp, fp, fn)

    return result


def multilabel_classification_report(
    preds: Tensor,
    target: Tensor,
    num_labels: int,
    threshold: float = 0.5,
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
    zero_division: float = 0.0,
) -> Dict[str, Dict[str, Tensor]]:
    r"""Compute a classification report for multilabel tasks.

    Generates per-label and average metrics (precision, recall, F1-score, support)
    similar to ``sklearn.metrics.classification_report`` but using PyTorch tensors.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, C, ...)``. If preds is a floating point
      tensor with values outside [0,1] range we consider the input to be logits and
      will auto apply sigmoid per element. Additionally, we convert to int tensor with
      thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, C, ...)``

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_labels: Integer specifying the number of labels
        threshold: Threshold for transforming probability to binary (0,1) predictions
        multidim_average:
            Defines how additionally dimensions ``...`` should be handled. Should be one of:

            - ``global``: Additional dimensions are flatted along the batch dimension
            - ``samplewise``: Statistic will be calculated independently for each sample

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        zero_division: Value to return when there is a zero division. Should be 0 or 1.

    Returns:
        A dictionary with per-label keys ``"label_0"``, ``"label_1"``, ..., ``"label_{L-1}"``
        and summary keys ``"micro"``, ``"macro"``, ``"weighted"``.
        Each value is a dict with keys ``"precision"``, ``"recall"``, ``"f1_score"``, ``"support"``.

    Example (preds is int tensor):
        >>> from torch import tensor
        >>> from torchmetrics.functional.classification import multilabel_classification_report
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0, 0, 1], [1, 0, 1]])
        >>> report = multilabel_classification_report(preds, target, num_labels=3)
        >>> report["label_0"]["precision"]
        tensor(1.)
        >>> report["label_1"]["recall"]
        tensor(0.)

    """
    _average = None

    if validate_args:
        _multilabel_stat_scores_arg_validation(num_labels, threshold, _average, multidim_average, ignore_index)
        _multilabel_stat_scores_tensor_validation(preds, target, num_labels, multidim_average, ignore_index)
    preds, target = _multilabel_stat_scores_format(preds, target, num_labels, threshold, ignore_index)
    tp, fp, tn, fn = _multilabel_stat_scores_update(preds, target, multidim_average)

    per_class = _compute_per_class_metrics(tp, fp, tn, fn, zero_division)
    result: Dict[str, Dict[str, Tensor]] = {}

    # Per-label entries
    for lbl_idx in range(num_labels):
        result[f"label_{lbl_idx}"] = {
            "precision": per_class["precision"][lbl_idx],
            "recall": per_class["recall"][lbl_idx],
            "f1_score": per_class["f1_score"][lbl_idx],
            "support": per_class["support"][lbl_idx],
        }

    # Summary averages
    result["micro"] = _compute_average(per_class, "micro", tp, fp, fn)
    result["macro"] = _compute_average(per_class, "macro", tp, fp, fn)
    result["weighted"] = _compute_average(per_class, "weighted", tp, fp, fn)

    return result
