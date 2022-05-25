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
from typing import Any, Dict, Optional

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.classification.confusion_matrix import ConfusionMatrix
from torchmetrics.functional.classification.jaccard import _jaccard_from_confmat


class JaccardIndex(ConfusionMatrix):
    r"""Computes Intersection over union, or `Jaccard index`_:

    .. math:: J(A,B) = \frac{|A\cap B|}{|A\cup B|}

    Where: :math:`A` and :math:`B` are both tensors of the same size, containing integer class values.
    They may be subject to conversion from input data (see description below). Note that it is different from box IoU.

    Works with binary, multiclass and multi-label data.
    Accepts probabilities from a model output or integer class values in prediction.
    Works with multi-dimensional preds and target.

    Forward accepts

    - ``preds`` (float or long tensor): ``(N, ...)`` or ``(N, C, ...)`` where C is the number of classes
    - ``target`` (long tensor): ``(N, ...)``

    If preds and target are the same shape and preds is a float tensor, we use the ``self.threshold`` argument
    to convert into integer labels. This is the case for binary and multi-label probabilities.

    If preds has an extra dimension as in the case of multi-class scores we perform an argmax on ``dim=1``.

    Args:
        num_classes: Number of classes in the dataset.
        ignore_index: optional int specifying a target class to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method. Has no effect if given an int that is not in the
            range [0, num_classes-1]. By default, no index is ignored, and all classes are used.
        absent_score: score to use for an individual class, if no instances of the class index were present in
            ``preds`` AND no instances of the class index were present in ``target``. For example, if we have 3 classes,
            [0, 0] for ``preds``, and [0, 2] for ``target``, then class 1 would be assigned the `absent_score`.
        threshold: Threshold value for binary or multi-label probabilities.
        multilabel: determines if data is multilabel or not.
        reduction: a method to reduce metric score over labels:

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from torchmetrics import JaccardIndex
        >>> target = torch.randint(0, 2, (10, 25, 25))
        >>> pred = torch.tensor(target)
        >>> pred[2:5, 7:13, 9:15] = 1 - pred[2:5, 7:13, 9:15]
        >>> jaccard = JaccardIndex(num_classes=2)
        >>> jaccard(pred, target)
        tensor(0.9660)

    """
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        num_classes: int,
        ignore_index: Optional[int] = None,
        absent_score: float = 0.0,
        threshold: float = 0.5,
        multilabel: bool = False,
        reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            normalize=None,
            threshold=threshold,
            multilabel=multilabel,
            **kwargs,
        )
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.absent_score = absent_score

    def compute(self) -> Tensor:
        """Computes intersection over union (IoU)"""
        return _jaccard_from_confmat(
            self.confmat, self.num_classes, self.ignore_index, self.absent_score, self.reduction
        )
