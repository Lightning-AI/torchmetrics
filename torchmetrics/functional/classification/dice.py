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
import math
from typing import Optional

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.classification.stat_scores import _reduce_stat_scores, _stat_scores_update
from torchmetrics.utilities.checks import _input_squeeze
from torchmetrics.utilities.enums import AverageMethod, MDMCAverageMethod
from torchmetrics.utilities.prints import rank_zero_warn


def dice_score(
    preds: Tensor,
    target: Tensor,
    bg: bool = False,
    nan_score: float = 0.0,
    no_fg_score: float = 0.0,
    reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
) -> Tensor:
    """Compute dice score from prediction scores.

    Supports only "macro" approach, which mean calculate the metric for each class separately,
    and average the metrics across classes (with equal weights for each class).

    .. deprecated:: v0.9
        The `dice_score` function was deprecated in v0.9 and will be removed in v0.10. Use `dice` function instead.

    Args:
        preds: estimated probabilities
        target: ground-truth labels
        bg: whether to also compute dice for the background
        nan_score: score to return, if a NaN occurs during computation
        no_fg_score: (default, ``0.0``) score to return, if no foreground pixel was found in target

            .. deprecated:: v0.9
                All different from default options will be changed to default.

        reduction: (default, ``'elementwise_mean'``) a method to reduce metric score over labels.

            .. deprecated:: v0.9
                All different from default options will be changed to default.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

    Return:
        Tensor containing dice score

    Example:
        >>> from torchmetrics.functional import dice_score
        >>> pred = torch.tensor([[0.85, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.85, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.85, 0.05],
        ...                      [0.05, 0.05, 0.05, 0.85]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> dice_score(pred, target)
        tensor(0.3333)
    """
    rank_zero_warn(
        "The `dice_score` function was deprecated in v0.9 and will be removed in v0.10. Use `dice` function instead.",
        DeprecationWarning,
    )
    num_classes = preds.shape[1]

    if no_fg_score != 0.0:
        no_fg_score = 0.0
        rank_zero_warn(f"Deprecated parameter. Switched to default `no_fg_score` = {no_fg_score}.")

    if reduction != "elementwise_mean":
        reduction = "elementwise_mean"
        rank_zero_warn(f"Deprecated parameter. Switched to default `reduction` = {reduction}.")

    zero_division = math.floor(nan_score)
    if zero_division != nan_score:
        rank_zero_warn(f"Deprecated parameter. `nan_score` converted to integer {zero_division}.")

    ignore_index = None
    if not bg:
        ignore_index = 0

    return dice(
        preds,
        target,
        ignore_index=ignore_index,
        average="macro",
        num_classes=num_classes,
        zero_division=zero_division,
    )


def _dice_compute(
    tp: Tensor,
    fp: Tensor,
    fn: Tensor,
    average: str,
    mdmc_average: Optional[str],
    zero_division: int = 0,
) -> Tensor:
    """Computes dice from the stat scores: true positives, false positives, false negatives.

    Args:
        tp: True positives
        fp: False positives
        fn: False negatives
        average: Defines the reduction that is applied
        mdmc_average: Defines how averaging is done for multi-dimensional multi-class inputs (on top of the
            ``average`` parameter)

    Example:
        >>> from torchmetrics.functional.classification.stat_scores import _stat_scores_update
        >>> from torchmetrics.functional.classification.dice import _dice_compute
        >>> preds  = torch.tensor([2, 0, 2, 1])
        >>> target = torch.tensor([1, 1, 2, 0])
        >>> tp, fp, tn, fn = _stat_scores_update(preds, target, reduce='micro')
        >>> _dice_compute(tp, fp, fn, average='micro', mdmc_average=None)
        tensor(0.2500)
    """
    numerator = 2 * tp
    denominator = 2 * tp + fp + fn

    if average == AverageMethod.MACRO and mdmc_average != MDMCAverageMethod.SAMPLEWISE:
        cond = tp + fp + fn == 0
        numerator = numerator[~cond]
        denominator = denominator[~cond]

    if average == AverageMethod.NONE and mdmc_average != MDMCAverageMethod.SAMPLEWISE:
        # a class is not present if there exists no TPs, no FPs, and no FNs
        meaningless_indeces = torch.nonzero((tp | fn | fp) == 0).cpu()
        numerator[meaningless_indeces, ...] = -1
        denominator[meaningless_indeces, ...] = -1

    return _reduce_stat_scores(
        numerator=numerator,
        denominator=denominator,
        weights=None if average != "weighted" else tp + fn,
        average=average,
        mdmc_average=mdmc_average,
        zero_division=zero_division,
    )


def dice(
    preds: Tensor,
    target: Tensor,
    zero_division: int = 0,
    average: str = "micro",
    mdmc_average: Optional[str] = "global",
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    num_classes: Optional[int] = None,
    multiclass: Optional[bool] = None,
    ignore_index: Optional[int] = None,
) -> Tensor:
    r"""Computes `Dice`_:

    .. math:: \text{Dice} = \frac{\text{2 * TP}}{\text{2 * TP} + \text{FP} + \text{FN}}

    Where :math:`\text{TP}` and :math:`\text{FN}` represent the number of true positives and
    false negatives respecitively.

    It is recommend set `ignore_index` to index of background class.

    The reduction method (how the recall scores are aggregated) is controlled by the
    ``average`` parameter, and additionally by the ``mdmc_average`` parameter in the
    multi-dimensional multi-class case. Accepts all inputs listed in :ref:`pages/classification:input types`.

    Args:
        preds: Predictions from model (probabilities, logits or labels)
        target: Ground truth values
        zero_division: The value to use for the score if denominator equals zero
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

            - ``None`` [default]: Should be left unchanged if your data is not multi-dimensional
              multi-class.

            - ``'samplewise'``: In this case, the statistics are computed separately for each
              sample on the ``N`` axis, and then averaged over samples.
              The computation for each sample is done by treating the flattened extra axes ``...``
              (see :ref:`pages/classification:input types`) as the ``N`` dimension within the sample,
              and computing the metric for the sample based on that.

            - ``'global'``: In this case the ``N`` and ``...`` dimensions of the inputs
              (see :ref:`pages/classification:input types`)
              are flattened into a new ``N_X`` sample axis, i.e. the inputs are treated as if they
              were ``(N_X, C)``. From here on the ``average`` parameter applies as usual.

        ignore_index:
            Integer specifying a target class to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method. If an index is ignored, and ``average=None``
            or ``'none'``, the score for the ignored class will be returned as ``nan``.

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

    Return:
        The shape of the returned tensor depends on the ``average`` parameter

        - If ``average in ['micro', 'macro', 'weighted', 'samples']``, a one-element tensor will be returned
        - If ``average in ['none', None]``, the shape will be ``(C,)``, where ``C`` stands  for the number of classes

    Raises:
        ValueError:
            If ``average`` is not one of ``"micro"``, ``"macro"``, ``"weighted"``, ``"samples"``, ``"none"`` or ``None``
        ValueError:
            If ``mdmc_average`` is not one of ``None``, ``"samplewise"``, ``"global"``.
        ValueError:
            If ``average`` is set but ``num_classes`` is not provided.
        ValueError:
            If ``num_classes`` is set and ``ignore_index`` is not in the range ``[0, num_classes)``.

    Example:
        >>> from torchmetrics.functional import dice
        >>> preds = torch.tensor([2, 0, 2, 1])
        >>> target = torch.tensor([1, 1, 2, 0])
        >>> dice(preds, target, average='micro')
        tensor(0.2500)

    """
    allowed_average = ("micro", "macro", "weighted", "samples", "none", None)
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
    reduce = "macro" if average in ("weighted", "none", None) else average

    tp, fp, _, fn = _stat_scores_update(
        preds,
        target,
        reduce=reduce,
        mdmc_reduce=mdmc_average,
        threshold=threshold,
        num_classes=num_classes,
        top_k=top_k,
        multiclass=multiclass,
        ignore_index=ignore_index,
    )

    return _dice_compute(tp, fp, fn, average, mdmc_average, zero_division)
