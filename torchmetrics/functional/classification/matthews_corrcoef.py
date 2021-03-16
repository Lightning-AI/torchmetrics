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

from torchmetrics.functional.classification.confusion_matrix import _confusion_matrix_update

def _matthews_corrcoef_update(
        preds: torch.Tensor,
        target: torch.Tensor,
        num_classes: int,
        threshold: float = 0.5
) -> torch.Tensor:
    return _confusion_matrix_update(preds, target, num_classes, threshold)
    
      
def _matthews_corrcoef_compute(confmat: torch.Tensor) -> torch.Tensor:
    tk = confmat.sum(dim=0)
    pk = confmat.sum(dim=1)
    c = torch.trace(confmat)
    s = confmat.sum()
    return (c*s - tk * pk) / (torch.sqrt(s**2 - pk*pk) * torch.sqrt(s**2 - tk*tk))

def matthews_corrcoef(
        preds: torch.Tensor,
        target: torch.Tensor,
        num_classes: int,
        weights: Optional[str] = None,
        threshold: float = 0.5
    ) -> torch.Tensor:
    r"""
    Calculates `Matthews correlation coefficient 
    <https://en.wikipedia.org/wiki/Matthews_correlation_coefficient>`_ that measures
    the general correlation or quality of a classification. It the binary case it
    is defined as:

    .. math::
        MCC = \frac{TP*TN - FP*FN}{\sqrt{(TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)}}

    where TP, TN, FP and FN are respectively the true postitives, true negatives,
    false positives and false negatives.

    Args:
        preds: (float or long tensor), Either a ``(N, ...)`` tensor with labels or
            ``(N, C, ...)`` where C is the number of classes, tensor with labels/probabilities
        target: ``target`` (long tensor), tensor with shape ``(N, ...)`` with ground true labels
        num_classes: Number of classes in the dataset.
        threshold:
            Threshold value for binary or multi-label probabilities. default: 0.5

    Example:
        >>> from torchmetrics.functional import cohen_kappa
        >>> target = torch.tensor([1, 1, 0, 0])
        >>> preds = torch.tensor([0, 1, 0, 0])
        >>> cohen_kappa(preds, target, num_classes=2)
        tensor(0.5000)

    """
    confmat = _matthews_corrcoef_update(preds, target, num_classes, threshold)
    return _matthews_corrcoef_compute(confmat)
