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
from torchmetrics.utilities.checks import _input_format_classification
from torchmetrics.utilities.compute import _safe_divide


def _hamming_distance_reduce(
    tp: Tensor,
    fp: Tensor,
    tn: Tensor,
    fn: Tensor,
    average: Optional[Literal["binary", "micro", "macro", "weighted", "none"]],
    multidim_average: Literal["global", "samplewise"] = "global",
) -> Tensor:
    if average == "binary" or average is None or average == "none":
        hamming_score = _safe_divide(tn, tn + fp)
    elif average == "micro":
        tn = tn.sum(dim=0 if multidim_average == "global" else 1)
        fp = fp.sum(dim=0 if multidim_average == "global" else 1)
        hamming_score = _safe_divide(tn, tn + fp)
    else:
        hamming_score = _safe_divide(tn, tn + fp)
        if average == "weighted":
            weights = tp + fn
        else:
            weights = torch.ones_like(hamming_score)
        hamming_score = _safe_divide(weights * hamming_score, weights.sum(-1, keepdim=True)).sum(-1)
    return 1 - hamming_score


def binary_hamming_distance(
    preds: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    if validate_args:
        _binary_stat_scores_arg_validation(threshold, multidim_average, ignore_index)
        _binary_stat_scores_tensor_validation(preds, target, multidim_average, ignore_index)
    preds, target = _binary_stat_scores_format(preds, target, threshold, ignore_index)
    tp, fp, tn, fn = _binary_stat_scores_update(preds, target, multidim_average)
    return _hamming_distance_reduce(tp, fp, tn, fn, average="binary", multidim_average=multidim_average)


def multiclass_hamming_distance(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "micro",
    top_k: int = 1,
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    if validate_args:
        _multiclass_stat_scores_arg_validation(num_classes, top_k, average, multidim_average, ignore_index)
        _multiclass_stat_scores_tensor_validation(preds, target, num_classes, multidim_average, ignore_index)
    preds, target = _multiclass_stat_scores_format(preds, target, top_k)
    tp, fp, tn, fn = _multiclass_stat_scores_update(preds, target, num_classes, top_k, multidim_average, ignore_index)
    return _hamming_distance_reduce(tp, fp, tn, fn, average=average, multidim_average=multidim_average)


def multilabel_hamming_distance(
    preds: Tensor,
    target: Tensor,
    num_labels: int,
    threshold: float = 0.5,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "micro",
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    if validate_args:
        _multilabel_stat_scores_arg_validation(num_labels, threshold, average, multidim_average, ignore_index)
        _multilabel_stat_scores_tensor_validation(preds, target, num_labels, multidim_average, ignore_index)
    preds, target = _multilabel_stat_scores_format(preds, target, num_labels, threshold, ignore_index)
    tp, fp, tn, fn = _multilabel_stat_scores_update(preds, target, multidim_average)
    return _hamming_distance_reduce(tp, fp, tn, fn, average=average, multidim_average=multidim_average)


# -------------------------- Old stuff --------------------------


def _hamming_distance_update(
    preds: Tensor,
    target: Tensor,
    threshold: float = 0.5,
) -> Tuple[Tensor, int]:
    """Returns the number of positions where prediction equals target, and number of predictions.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        threshold: Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case
            of binary or multi-label inputs. Default value of 0.5 corresponds to input being probabilities.
    """

    preds, target, _ = _input_format_classification(preds, target, threshold=threshold)

    correct = (preds == target).sum()
    total = preds.numel()

    return correct, total


def _hamming_distance_compute(correct: Tensor, total: Union[int, Tensor]) -> Tensor:
    """Computes the Hamming distance.

    Args:
        correct: Number of positions where prediction equals target
        total: Total number of predictions

    Example:
        >>> target = torch.tensor([[0, 1], [1, 1]])
        >>> preds = torch.tensor([[0, 1], [0, 1]])
        >>> correct, total = _hamming_distance_update(preds, target)
        >>> _hamming_distance_compute(correct, total)
        tensor(0.2500)
    """

    return 1 - correct.float() / total


def hamming_distance(preds: Tensor, target: Tensor, threshold: float = 0.5) -> Tensor:
    r"""
    Computes the average `Hamming distance`_ (also
    known as Hamming loss) between targets and predictions:

    .. math::
        \text{Hamming distance} = \frac{1}{N \cdot L} \sum_i^N \sum_l^L 1(y_{il} \neq \hat{y}_{il})

    Where :math:`y` is a tensor of target values, :math:`\hat{y}` is a tensor of predictions,
    and :math:`\bullet_{il}` refers to the :math:`l`-th label of the :math:`i`-th sample of that
    tensor.

    This is the same as ``1-accuracy`` for binary data, while for all other types of inputs it
    treats each possible label separately - meaning that, for example, multi-class data is
    treated as if it were multi-label.

    Accepts all input types listed in :ref:`pages/classification:input types`.

    Args:
        preds: Predictions from model (probabilities, logits or labels)
        target: Ground truth
        threshold:
            Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case
            of binary or multi-label inputs. Default value of 0.5 corresponds to input being probabilities.

    Example:
        >>> from torchmetrics.functional import hamming_distance
        >>> target = torch.tensor([[0, 1], [1, 1]])
        >>> preds = torch.tensor([[0, 1], [0, 1]])
        >>> hamming_distance(preds, target)
        tensor(0.2500)
    """

    correct, total = _hamming_distance_update(preds, target, threshold)
    return _hamming_distance_compute(correct, total)
