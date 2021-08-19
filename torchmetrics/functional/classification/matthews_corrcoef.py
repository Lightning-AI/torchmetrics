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
import torch
from torch import Tensor

from torchmetrics.functional.classification.confusion_matrix import _confusion_matrix_update

_matthews_corrcoef_update = _confusion_matrix_update


def _matthews_corrcoef_compute(confmat: Tensor) -> Tensor:
    """Computes Matthews correlation coefficient.

    Args:
        confmat: Confusion matrix

    Example:
        >>> target = torch.tensor([1, 1, 0, 0])
        >>> preds = torch.tensor([0, 1, 0, 0])
        >>> confmat = _matthews_corrcoef_update(preds, target, num_classes=2)
        >>> _matthews_corrcoef_compute(confmat)
        tensor(0.5774)
    """

    tk = confmat.sum(dim=1).float()
    pk = confmat.sum(dim=0).float()
    c = torch.trace(confmat).float()
    s = confmat.sum().float()
    return (c * s - sum(tk * pk)) / (torch.sqrt(s ** 2 - sum(pk * pk)) * torch.sqrt(s ** 2 - sum(tk * tk)))


def matthews_corrcoef(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    threshold: float = 0.5,
) -> Tensor:
    r"""
    Calculates `Matthews correlation coefficient`_ that measures
    the general correlation or quality of a classification. In the binary case it
    is defined as:

    .. math::
        MCC = \frac{TP*TN - FP*FN}{\sqrt{(TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)}}

    where TP, TN, FP and FN are respectively the true postitives, true negatives,
    false positives and false negatives. Also works in the case of multi-label or
    multi-class input.

    Args:
        preds: (float or long tensor), Either a ``(N, ...)`` tensor with labels or
            ``(N, C, ...)`` where C is the number of classes, tensor with labels/probabilities
        target: ``target`` (long tensor), tensor with shape ``(N, ...)`` with ground true labels
        num_classes: Number of classes in the dataset.
        threshold:
            Threshold value for binary or multi-label probabilities. default: 0.5

    Example:
        >>> from torchmetrics.functional import matthews_corrcoef
        >>> target = torch.tensor([1, 1, 0, 0])
        >>> preds = torch.tensor([0, 1, 0, 0])
        >>> matthews_corrcoef(preds, target, num_classes=2)
        tensor(0.5774)

    """
    confmat = _matthews_corrcoef_update(preds, target, num_classes, threshold)
    return _matthews_corrcoef_compute(confmat)
