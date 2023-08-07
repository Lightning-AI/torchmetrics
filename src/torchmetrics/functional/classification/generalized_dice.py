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
from typing_extensions import Any, Literal

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
from torchmetrics.utilities.compute import _safe_divide
from torchmetrics.utilities.enums import AverageMethod as AvgMethod
from torchmetrics.utilities.enums import MDMCAverageMethod
from torchmetrics.utilities.prints import rank_zero_warn


def _generalized_dice_reduce(
    tp: Tensor,
    fp: Tensor,
    tn: Tensor,
    fn: Tensor,
    weight_type: Optional[Literal["square", "simple"]],
    average: Optional[Literal["binary", "micro", "macro", "weighted", "none"]],
    multidim_average: Literal["global", "samplewise"] = "global",
) -> Tensor:
    target_volume = tp + fn
    if weight_type == "simple":
        weights = torch.reciprocal(target_volume.float())
    elif weight_type == "square":
        weights = torch.reciprocal(target_volume.float() * target_volume.float())
    elif weight_type is None:
        weights = torch.ones_like(target_volume.float())

    if weights.ndim > 1:
        for sample_weights in weights:
            infs = torch.isinf(sample_weights)
            sample_weights[infs] = torch.max(sample_weights[~infs]) if len(sample_weights[~infs]) > 0 else 0
    else:
        infs = torch.isinf(weights)
        weights[infs] = torch.max(weights[~infs])

    if average == "binary":
        return _safe_divide(2 * (tp + weights), 2 * tp + fp + fn)
    elif average == "micro":
        tp = tp.sum(dim=0 if multidim_average == "global" else 1)
        fn = fn.sum(dim=0 if multidim_average == "global" else 1)
        fp = fp.sum(dim=0 if multidim_average == "global" else 1)
        weights = weights.sum(dim=0 if multidim_average == "global" else 1)
        return _safe_divide(2 * (tp + weights), 2 * tp + fp + fn)
    else:
        generalized_dice_score = _safe_divide(2 * (tp + weights), 2 * tp + fp + fn)
        if average is None or average == "none":
            return generalized_dice_score
        weights = tp + fn if average == "weighted" else torch.ones_like(generalized_dice_score)
        return _safe_divide(weights * generalized_dice_score, weights.sum(-1, keepdim=True)).sum(-1)


def _binary_generalized_dice_score_arg_validation(
    weight_type: Optional[Literal["square", "simple"]],
    threshold: float = 0.5,
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: Optional[int] = None,
) -> None:
    allowed_weight_type = ("square", "simple", None)
    if weight_type not in weight_type:
        raise ValueError(
            f"Argument `weight_type` needs to one of the following: {allowed_weight_type} but got {weight_type}"
        )
    _binary_stat_scores_arg_validation(threshold, multidim_average, ignore_index)


def binary_generalized_dice_score(
    preds: Tensor,
    target: Tensor,
    weight_type: Optional[Literal["square", "simple"]],
    threshold: float = 0.5,
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    if validate_args:
        _binary_generalized_dice_score_arg_validation(weight_type, threshold, multidim_average, ignore_index)
        _binary_stat_scores_tensor_validation(preds, target, multidim_average, ignore_index)
    preds, target = _binary_stat_scores_format(preds, target, threshold, ignore_index)
    tp, fp, tn, fn = _binary_stat_scores_update(preds, target, multidim_average)
    return _generalized_dice_reduce(tp, fp, tn, fn, weight_type, average="binary", multidim_average=multidim_average)


def _multiclass_generalized_dice_score_arg_validation(
    weight_type: Optional[Literal["square", "simple"]],
    num_classes: int,
    top_k: int = 1,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: Optional[int] = None,
) -> None:
    allowed_weight_type = ("square", "simple", None)
    if weight_type not in weight_type:
        raise ValueError(
            f"Argument `weight_type` needs to one of the following: {allowed_weight_type} but got {weight_type}"
        )
    _multiclass_stat_scores_arg_validation(num_classes, top_k, average, multidim_average, ignore_index)


def multiclass_generalized_dice_score(
    preds: Tensor,
    target: Tensor,
    weight_type: Optional[Literal["square", "simple"]],
    num_classes: int,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
    top_k: int = 1,
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    if validate_args:
        _multiclass_generalized_dice_score_arg_validation(
            weight_type, num_classes, top_k, average, multidim_average, ignore_index
        )
        _multiclass_stat_scores_tensor_validation(preds, target, num_classes, multidim_average, ignore_index)
    preds, target = _multiclass_stat_scores_format(preds, target, top_k)
    tp, fp, tn, fn = _multiclass_stat_scores_update(
        preds, target, num_classes, top_k, average, multidim_average, ignore_index
    )
    return _generalized_dice_reduce(tp, fp, tn, fn, weight_type, average=average, multidim_average=multidim_average)


def _multilabel_generalized_dice_score_arg_validation(
    weight_type: Optional[Literal["square", "simple"]],
    num_labels: int,
    threshold: float = 0.5,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: Optional[int] = None,
) -> None:
    allowed_weight_type = ("square", "simple", None)
    if weight_type not in weight_type:
        raise ValueError(
            f"Argument `weight_type` needs to one of the following: {allowed_weight_type} but got {weight_type}"
        )
    _multilabel_stat_scores_arg_validation(num_labels, threshold, average, multidim_average, ignore_index)


def multilabel_generalized_dice_score(
    preds: Tensor,
    target: Tensor,
    weight_type: Optional[Literal["square", "simple"]],
    num_labels: int,
    threshold: float = 0.5,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    if validate_args:
        _multilabel_generalized_dice_score_arg_validation(
            weight_type, num_labels, threshold, average, multidim_average, ignore_index
        )
        _multilabel_stat_scores_tensor_validation(preds, target, num_labels, multidim_average, ignore_index)
    preds, target = _multilabel_stat_scores_format(preds, target, num_labels, threshold, ignore_index)
    tp, fp, tn, fn = _multilabel_stat_scores_update(preds, target, multidim_average)
    return _generalized_dice_reduce(tp, fp, tn, fn, weight_type, average=average, multidim_average=multidim_average)


def _generalized_dice_compute(
    tp: Tensor,
    fp: Tensor,
    fn: Tensor,
    average: str = "samples",
    weight_type: str = "square",
    ignore_index: Optional[int] = None,
    zero_division: Optional[int] = None,
) -> Tensor:
    """Computes generalized dice score from the stat scores: true positives, false positives, false negatives.

    Args:
        tp: True positives
        fp: False positives
        fn: False negatives
        average: Defines the reduction that is applied
        weight_type: Defines the type of weights applied different classes
        ignore_index: Optional index of the class to ignore in the score computation
        zero_division: The value to use for the score if denominator equals zero. If set to 0, score will be 1
            if the numerator is also 0, and 0 otherwise
    """
    # Compute ground-truth class volume and class weights
    target_volume = tp + fn
    if weight_type == "simple":
        weights = torch.reciprocal(target_volume.float())
    elif weight_type == "square":
        weights = torch.reciprocal(target_volume.float() * target_volume.float())
    elif weight_type is None:
        weights = torch.ones_like(target_volume.float())

    # Replace weights and stats for ignore_index by 0
    if ignore_index is not None:
        weights[..., ignore_index] = 0
        tp[..., ignore_index] = 0
        fp[..., ignore_index] = 0
        fn[..., ignore_index]

    # Replace infinite weights for non-appearing classes by the max« weight or 0, if all weights are infinite
    if weights.dim() > 1:
        for sample_weights in weights:
            infs = torch.isinf(sample_weights)
            sample_weights[infs] = torch.max(sample_weights[~infs]) if len(sample_weights[~infs]) > 0 else 0
    else:
        infs = torch.isinf(weights)
        weights[infs] = torch.max(weights[~infs])

    # Compute weighted numerator and denominator
    numerator = 2 * (tp * weights).sum(dim=-1)
    denominator = ((2 * tp + fp + fn) * weights).sum(dim=-1)

    # Handle zero division
    denominator_zeros = denominator == 0
    denominator[denominator_zeros] = 1
    if zero_division is not None:
        # If zero_division score is specified, use it as numerator and set denominator to 1
        numerator[denominator_zeros] = zero_division
    else:
        # If both denominator and total sample prediction volume are 0, score is 1. Otherwise 0.
        pred_volume = (tp + fp).sum(dim=-1)
        pred_zeros = pred_volume == 0
        numerator[denominator_zeros] = torch.where(
            pred_zeros[denominator_zeros],
            torch.tensor(1, device=numerator.device).float(),
            torch.tensor(0, device=numerator.device).float(),
        )

    return _reduce_stat_scores(
        numerator=numerator,
        denominator=denominator,
        weights=None,
        average=average,
        mdmc_average=None,
    )


def generalized_dice_score(
    preds: Tensor,
    target: Tensor,
    weight_type: str = "square",
    zero_division: Optional[int] = None,
    average: str = "samples",
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    num_classes: Optional[int] = None,
    multiclass: bool = True,
    multidim: bool = True,
    ignore_index: Optional[int] = None,
    **kwargs: Any,
) -> Tensor:
    r"""Computes the Generalized Dice Score (GDS) metric:

    .. math::
        \text{GDS}=\sum_{i=1}^{C}\frac{2\cdot\text{TP}_i}{(2\cdot\text{TP}_i+\text{FP}_i+\text{FN}_i)\cdot w_i}

    Where :math:`\text{C}` is the number of classes and :math:`\text{TP}_i`, :math:`\text{FP}_i` and :math:`\text{FN}`_i
    represent the numbers of true positives, false positives and false negatives for class :math:`i`, respectively.
    :math:`w_i` represents the weight of class :math:`i`.

    The reduction method (how the recall scores are aggregated) is controlled by the
    ``average`` parameter. Accepts all inputs listed in :ref:`pages/classification:input types`.

    Args:
        preds: Predictions from model (probabilities, logits or labels).

        target: Ground truth values.

        weight_type: Defines the type of weighting to apply. Should be one of the following:

            - ``'square'`` [default]: Weight each class by the squared inverse of its support,
              i.e., the inverse of its squared volume - :math:`\frac{1}{(tp + fn)^2}`.
            - ``'simple'``: Weight each class by the inverse of its support, i.e.,
              the inverse of its volume - :math:`\frac{1}{tp + fn}`.
            - ``None``: All classes are assigned unitary weight. Equivalent to dice score.

        zero_division:
            The value to use for the score if denominator equals zero. If set to None, the score will be 1 if the
            numerator is also 0, and 0 otherwise.

        average:
            Defines the reduction that is applied. Should be one of the following:

            - ``'samples'`` [default]: Calculate the metric for each sample, and average the metrics
              across samples (with equal weights for each sample).
            - ``'none'`` or ``None``: Calculate the metric for each sample separately, and return
              the metric for every sample.

        threshold:
            Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case
            of binary or multi-label inputs. Default value of 0.5 corresponds to input being probabilities.

        top_k:
            Number of the highest probability or logit score predictions considered finding the correct label.
            The default value (``None``) will be interpreted as 1.

        num_classes:
            Number of classes.

        multiclass:
            Determines whether the input is multiclass (if True) or multilabel (if False).

        multidim:
            Determines whether the input is multidim or not.

        ignore_index:
            Integer specifying a target class to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method.

    Return:
        The shape of the returned tensor depends on the ``average`` parameter

        - If ``average == 'samples'``, a one-element tensor will be returned
        - If ``average in ['none', None]``, the shape will be ``(N,)``, where ``N`` stands  for the number of samples

    Raises:
        ValueError:
            If ``weight_type`` is not ``"simple"``, ``"square"`` or ``None``.
        ValueError:
            If ``average`` is not one of ``"samples"``, ``"none"`` or ``None``.
        ValueError:
            If ``num_classes`` is provided but is not an integer larger than 0.
        ValueError:
            If ``num_classes`` is set and ``ignore_index`` is not in the range ``[0, num_classes)``.
        ValueError:
            If ``top_k`` is not an integer larger than ``0``.

    Example:
        >>> from torch import tensor
        >>> from torchmetrics.functional import generalized_dice_score
        >>> preds = tensor([2, 0, 2, 1])
        >>> target = tensor([1, 1, 2, 0])
        >>> generalized_dice_score(preds, target, average='samples')
        tensor(0.3478)

    """
    allowed_weight_type = ("square", "simple", None)
    if weight_type not in allowed_weight_type:
        raise ValueError(f"The `weight_type` has to be one of {allowed_weight_type}, got {weight_type}.")

    allowed_average = ("samples", "none", None)
    if average not in allowed_average:
        raise ValueError(f"The `average` has to be one of {allowed_average}, got {average}.")

    if num_classes and num_classes < 1:
        raise ValueError("Number of classes must be larger than 0.")

    if num_classes and ignore_index is not None and (not ignore_index < num_classes or num_classes == 1):
        raise ValueError(f"The `ignore_index` {ignore_index} is not valid for inputs with {num_classes} classes")

    if top_k is not None and (not isinstance(top_k, int) or top_k <= 0):
        raise ValueError(f"The `top_k` should be an integer larger than 0, got {top_k}")

    preds, target = _input_squeeze(preds, target)

    # Obtain tp, fp and fn per sample per class
    reduce = "macro" if multidim else None
    tp, fp, _, fn = _stat_scores_update(
        preds,
        target,
        reduce=reduce,
        mdmc_reduce="samplewise",
        threshold=threshold,
        num_classes=num_classes,
        top_k=top_k,
        multiclass=multiclass,
        ignore_index=ignore_index,
    )

    return _generalized_dice_compute(
        tp,
        fp,
        fn,
        average=average,
        ignore_index=None if reduce is None else ignore_index,
        weight_type=weight_type,
        zero_division=zero_division,
    )
