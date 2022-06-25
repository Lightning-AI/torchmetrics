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

from torchmetrics.functional.classification.confusion_matrix import _confusion_matrix_update


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
        scores[union == 0] = absent_score

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
    average: Optional[str] = "macro",
    ignore_index: Optional[int] = None,
    absent_score: float = 0.0,
    threshold: float = 0.5,
) -> Tensor:
    r"""Computes `Jaccard index`_

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

    confmat = _confusion_matrix_update(preds, target, num_classes, threshold)
    return _jaccard_from_confmat(confmat, num_classes, average, ignore_index, absent_score)
