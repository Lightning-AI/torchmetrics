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
from typing_extensions import Literal

from torchmetrics.functional.classification.confusion_matrix import (
    _binary_confusion_matrix_arg_validation,
    _binary_confusion_matrix_format,
    _binary_confusion_matrix_tensor_validation,
    _binary_confusion_matrix_update,
    _confusion_matrix_update,
    _multiclass_confusion_matrix_arg_validation,
    _multiclass_confusion_matrix_format,
    _multiclass_confusion_matrix_tensor_validation,
    _multiclass_confusion_matrix_update,
    _multilabel_confusion_matrix_arg_validation,
    _multilabel_confusion_matrix_format,
    _multilabel_confusion_matrix_tensor_validation,
    _multilabel_confusion_matrix_update,
)
from torchmetrics.utilities.compute import _safe_divide
from torchmetrics.utilities.prints import rank_zero_warn


def _jaccard_index_reduce(
    confmat: Tensor,
    average: Optional[Literal["micro", "macro", "weighted", "none", "binary"]],
) -> Tensor:
    """Perform reduction of an un-normalized confusion matrix into jaccard score.

    Args:
        confmat: tensor with un-normalized confusionmatrix
        average: reduction method

            - ``'binary'``: binary reduction, expects a 2x2 matrix
            - ``'macro'``: Calculate the metric for each class separately, and average the
              metrics across classes (with equal weights for each class).
            - ``'micro'``: Calculate the metric globally, across all samples and classes.
            - ``'weighted'``: Calculate the metric for each class separately, and average the
              metrics across classes, weighting each class by its support (``tp + fn``).
            - ``'none'`` or ``None``: Calculate the metric for each class separately, and return
              the metric for every class.
    """
    allowed_average = ["binary", "micro", "macro", "weighted", "none", None]
    if average not in allowed_average:
        raise ValueError(f"The `average` has to be one of {allowed_average}, got {average}.")
    confmat = confmat.float()
    if average == "binary":
        return confmat[1, 1] / (confmat[0, 1] + confmat[1, 0] + confmat[1, 1])
    else:
        if confmat.ndim == 3:  # multilabel
            num = confmat[:, 1, 1]
            denom = confmat[:, 1, 1] + confmat[:, 0, 1] + confmat[:, 1, 0]
        else:  # multiclass
            num = torch.diag(confmat)
            denom = confmat.sum(0) + confmat.sum(1) - num

        if average == "micro":
            num = num.sum()
            denom = denom.sum()

        jaccard = _safe_divide(num, denom)

        if average is None or average == "none":
            return jaccard
        if average == "weighted":
            weights = confmat[:, 1, 1] + confmat[:, 1, 0] if confmat.ndim == 3 else confmat.sum(1)
        else:
            weights = torch.ones_like(jaccard)
        return ((weights * jaccard) / weights.sum()).sum()


def binary_jaccard_index(
    preds: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    r"""Calculates the Jaccard index for binary tasks. The `Jaccard index`_ (also known as the intersetion over
    union or jaccard similarity coefficient) is an statistic that can be used to determine the similarity and
    diversity of a sample set. It is defined as the size of the intersection divided by the union of the sample
    sets:

    .. math:: J(A,B) = \frac{|A\cap B|}{|A\cup B|}

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Addtionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, ...)``

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        threshold: Threshold for transforming probability to binary (0,1) predictions
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        normalize: Normalization mode for confusion matrix. Choose from:

            - ``None`` or ``'none'``: no normalization (default)
            - ``'true'``: normalization over the targets (most commonly used)
            - ``'pred'``: normalization over the predictions
            - ``'all'``: normalization over the whole matrix

        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (preds is int tensor):
        >>> from torchmetrics.functional.classification import binary_jaccard_index
        >>> target = torch.tensor([1, 1, 0, 0])
        >>> preds = torch.tensor([0, 1, 0, 0])
        >>> binary_jaccard_index(preds, target)
        tensor(0.5000)

    Example (preds is float tensor):
        >>> from torchmetrics.functional.classification import binary_jaccard_index
        >>> target = torch.tensor([1, 1, 0, 0])
        >>> preds = torch.tensor([0.35, 0.85, 0.48, 0.01])
        >>> binary_jaccard_index(preds, target)
        tensor(0.5000)
    """
    if validate_args:
        _binary_confusion_matrix_arg_validation(threshold, ignore_index)
        _binary_confusion_matrix_tensor_validation(preds, target, ignore_index)
    preds, target = _binary_confusion_matrix_format(preds, target, threshold, ignore_index)
    confmat = _binary_confusion_matrix_update(preds, target)
    return _jaccard_index_reduce(confmat, average="binary")


def _multiclass_jaccard_index_arg_validation(
    num_classes: int,
    ignore_index: Optional[int] = None,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = None,
) -> None:
    _multiclass_confusion_matrix_arg_validation(num_classes, ignore_index)
    allowed_average = ("micro", "macro", "weighted", "none", None)
    if average not in allowed_average:
        raise ValueError(f"Expected argument `average` to be one of {allowed_average}, but got {average}.")


def multiclass_jaccard_index(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    r"""Calculates the Jaccard index for multiclass tasks. The `Jaccard index`_ (also known as the intersetion over
    union or jaccard similarity coefficient) is an statistic that can be used to determine the similarity and
    diversity of a sample set. It is defined as the size of the intersection divided by the union of the sample
    sets:

    .. math:: J(A,B) = \frac{|A\cap B|}{|A\cup B|}

    Accepts the following input tensors:

    - ``preds``: ``(N, ...)`` (int tensor) or ``(N, C, ..)`` (float tensor). If preds is a floating point
      we apply ``torch.argmax`` along the ``C`` dimension to automatically convert probabilities/logits into
      an int tensor.
    - ``target`` (int tensor): ``(N, ...)``

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        num_classes: Integer specifing the number of classes
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: Calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: Calculates statistic for each label and applies no reduction

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (pred is integer tensor):
        >>> from torchmetrics.functional.classification import multiclass_jaccard_index
        >>> target = torch.tensor([2, 1, 0, 0])
        >>> preds = torch.tensor([2, 1, 0, 1])
        >>> multiclass_jaccard_index(preds, target, num_classes=3)
        tensor(0.6667)

    Example (pred is float tensor):
        >>> from torchmetrics.functional.classification import multiclass_jaccard_index
        >>> target = torch.tensor([2, 1, 0, 0])
        >>> preds = torch.tensor([
        ...   [0.16, 0.26, 0.58],
        ...   [0.22, 0.61, 0.17],
        ...   [0.71, 0.09, 0.20],
        ...   [0.05, 0.82, 0.13],
        ... ])
        >>> multiclass_jaccard_index(preds, target, num_classes=3)
        tensor(0.6667)
    """
    if validate_args:
        _multiclass_jaccard_index_arg_validation(num_classes, ignore_index, average)
        _multiclass_confusion_matrix_tensor_validation(preds, target, num_classes, ignore_index)
    preds, target = _multiclass_confusion_matrix_format(preds, target, ignore_index)
    confmat = _multiclass_confusion_matrix_update(preds, target, num_classes)
    return _jaccard_index_reduce(confmat, average=average)


def _multilabel_jaccard_index_arg_validation(
    num_labels: int,
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
) -> None:
    _multilabel_confusion_matrix_arg_validation(num_labels, threshold, ignore_index)
    allowed_average = ("micro", "macro", "weighted", "none", None)
    if average not in allowed_average:
        raise ValueError(f"Expected argument `average` to be one of {allowed_average}, but got {average}.")


def multilabel_jaccard_index(
    preds: Tensor,
    target: Tensor,
    num_labels: int,
    threshold: float = 0.5,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    r"""Calculates the Jaccard index for multilabel tasks. The `Jaccard index`_ (also known as the intersetion over
    union or jaccard similarity coefficient) is an statistic that can be used to determine the similarity and
    diversity of a sample set. It is defined as the size of the intersection divided by the union of the sample
    sets:

    .. math:: J(A,B) = \frac{|A\cap B|}{|A\cup B|}

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, C, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Addtionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, C, ...)``

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        num_classes: Integer specifing the number of labels
        threshold: Threshold for transforming probability to binary (0,1) predictions
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: Calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: Calculates statistic for each label and applies no reduction

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (preds is int tensor):
        >>> from torchmetrics.functional.classification import multilabel_jaccard_index
        >>> target = torch.tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = torch.tensor([[0, 0, 1], [1, 0, 1]])
        >>> multilabel_jaccard_index(preds, target, num_labels=3)
        tensor(0.5000)

    Example (preds is float tensor):
        >>> from torchmetrics.functional.classification import multilabel_jaccard_index
        >>> target = torch.tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = torch.tensor([[0.11, 0.22, 0.84], [0.73, 0.33, 0.92]])
        >>> multilabel_jaccard_index(preds, target, num_labels=3)
        tensor(0.5000)
    """
    if validate_args:
        _multilabel_jaccard_index_arg_validation(num_labels, threshold, ignore_index)
        _multilabel_confusion_matrix_tensor_validation(preds, target, num_labels, ignore_index)
    preds, target = _multilabel_confusion_matrix_format(preds, target, num_labels, threshold, ignore_index)
    confmat = _multilabel_confusion_matrix_update(preds, target, num_labels)
    return _jaccard_index_reduce(confmat, average=average)


def _jaccard_from_confmat(
    confmat: Tensor,
    num_classes: int,
    average: Optional[str] = "macro",
    ignore_index: Optional[int] = None,
    absent_score: float = 0.0,
) -> Tensor:
    """Computes the intersection over union from confusion matrix.

    Args:
        confmat: Confusion matrix without normalization
        num_classes: Number of classes for a given prediction and target tensor
        average:
            Defines the reduction that is applied. Should be one of the following:

            - ``'macro'`` [default]: Calculate the metric for each class separately, and average the
              metrics across classes (with equal weights for each class).
            - ``'micro'``: Calculate the metric globally, across all samples and classes.
            - ``'weighted'``: Calculate the metric for each class separately, and average the
              metrics across classes, weighting each class by its support (``tp + fn``).
            - ``'none'`` or ``None``: Calculate the metric for each class separately, and return
              the metric for every class. Note that if a given class doesn't occur in the
              `preds` or `target`, the value for the class will be ``nan``.

        ignore_index: optional int specifying a target class to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method.
        absent_score: score to use for an individual class, if no instances of the class index were present in `pred`
            AND no instances of the class index were present in `target`.
    """
    allowed_average = ["micro", "macro", "weighted", "none", None]
    if average not in allowed_average:
        raise ValueError(f"The `average` has to be one of {allowed_average}, got {average}.")

    # Remove the ignored class index from the scores.
    if ignore_index is not None and 0 <= ignore_index < num_classes:
        confmat[ignore_index] = 0.0

    if average == "none" or average is None:
        intersection = torch.diag(confmat)
        union = confmat.sum(0) + confmat.sum(1) - intersection

        # If this class is absent in both target AND pred (union == 0), then use the absent_score for this class.
        scores = intersection.float() / union.float()
        scores = scores.where(union != 0, torch.tensor(absent_score, dtype=scores.dtype, device=scores.device))

        if ignore_index is not None and 0 <= ignore_index < num_classes:
            scores = torch.cat(
                [
                    scores[:ignore_index],
                    scores[ignore_index + 1 :],
                ]
            )
        return scores

    if average == "macro":
        scores = _jaccard_from_confmat(
            confmat, num_classes, average="none", ignore_index=ignore_index, absent_score=absent_score
        )
        return torch.mean(scores)

    if average == "micro":
        intersection = torch.sum(torch.diag(confmat))
        union = torch.sum(torch.sum(confmat, dim=1) + torch.sum(confmat, dim=0) - torch.diag(confmat))
        return intersection.float() / union.float()

    weights = torch.sum(confmat, dim=1).float() / torch.sum(confmat).float()
    scores = _jaccard_from_confmat(
        confmat, num_classes, average="none", ignore_index=ignore_index, absent_score=absent_score
    )
    return torch.sum(weights * scores)


def jaccard_index(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
    ignore_index: Optional[int] = None,
    absent_score: float = 0.0,
    threshold: float = 0.5,
    task: Optional[Literal["binary", "multiclass", "multilabel"]] = None,
    num_labels: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    r"""Jaccard index.

    .. note::
        From v0.10 an ``'binary_*'``, ``'multiclass_*'``, ``'multilabel_*'`` version now exist of each classification
        metric. Moving forward we recommend using these versions. This base metric will still work as it did
        prior to v0.10 until v0.11. From v0.11 the `task` argument introduced in this metric will be required
        and the general order of arguments may change, such that this metric will just function as an single
        entrypoint to calling the three specialized versions.

    Computes `Jaccard index`_

    .. math:: J(A,B) = \frac{|A\cap B|}{|A\cup B|}

    Where: :math:`A` and :math:`B` are both tensors of the same size,
    containing integer class values. They may be subject to conversion from
    input data (see description below).

    Note that it is different from box IoU.

    If preds and target are the same shape and preds is a float tensor, we use the ``self.threshold`` argument
    to convert into integer labels. This is the case for binary and multi-label probabilities.

    If pred has an extra dimension as in the case of multi-class scores we
    perform an argmax on ``dim=1``.

    Args:
        preds: tensor containing predictions from model (probabilities, or labels) with shape ``[N, d1, d2, ...]``
        target: tensor containing ground truth labels with shape ``[N, d1, d2, ...]``
        num_classes: Specify the number of classes
        average:
            Defines the reduction that is applied. Should be one of the following:

            - ``'macro'`` [default]: Calculate the metric for each class separately, and average the
              metrics across classes (with equal weights for each class).
            - ``'micro'``: Calculate the metric globally, across all samples and classes.
            - ``'weighted'``: Calculate the metric for each class separately, and average the
              metrics across classes, weighting each class by its support (``tp + fn``).
            - ``'none'`` or ``None``: Calculate the metric for each class separately, and return
              the metric for every class. Note that if a given class doesn't occur in the
              `preds` or `target`, the value for the class will be ``nan``.

        ignore_index: optional int specifying a target class to ignore. If given,
            this class index does not contribute to the returned score, regardless
            of reduction method. Has no effect if given an int that is not in the
            range ``[0, num_classes-1]``, where num_classes is either given or derived
            from pred and target. By default, no index is ignored, and all classes are used.
        absent_score: score to use for an individual class, if no instances of
            the class index were present in ``preds`` AND no instances of the class
            index were present in ``target``. For example, if we have 3 classes,
            [0, 0] for ``preds``, and [0, 2] for ``target``, then class 1 would be
            assigned the `absent_score`.
        threshold: Threshold value for binary or multi-label probabilities.

    Return:
        The shape of the returned tensor depends on the ``average`` parameter

        - If ``average in ['micro', 'macro', 'weighted']``, a one-element tensor will be returned
        - If ``average in ['none', None]``, the shape will be ``(C,)``, where ``C`` stands  for the number
          of classes

    Example:
        >>> from torchmetrics.functional import jaccard_index
        >>> target = torch.randint(0, 2, (10, 25, 25))
        >>> pred = torch.tensor(target)
        >>> pred[2:5, 7:13, 9:15] = 1 - pred[2:5, 7:13, 9:15]
        >>> jaccard_index(pred, target, num_classes=2)
        tensor(0.9660)
    """
    if task is not None:
        if task == "binary":
            return binary_jaccard_index(preds, target, threshold, ignore_index, validate_args)
        if task == "multiclass":
            assert isinstance(num_classes, int)
            return multiclass_jaccard_index(preds, target, num_classes, average, ignore_index, validate_args)
        if task == "multilabel":
            assert isinstance(num_labels, int)
            return multilabel_jaccard_index(preds, target, num_labels, threshold, average, ignore_index, validate_args)
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
    confmat = _confusion_matrix_update(preds, target, num_classes, threshold)
    return _jaccard_from_confmat(confmat, num_classes, average, ignore_index, absent_score)
