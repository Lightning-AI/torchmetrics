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
from typing import Any, Optional

import torch
from deprecate import deprecated, void

from torchmetrics.classification.jaccard import JaccardIndex


class IoU(JaccardIndex):
    r"""
    Computes Intersection over union, or `Jaccard index`_:

    .. deprecated:: v0.7
        Use :class:`torchmetrics.JaccardIndex`. Will be removed in v0.8.

    Example:
        >>> from torchmetrics import IoU
        >>> target = torch.randint(0, 2, (10, 25, 25))
        >>> pred = torch.tensor(target)
        >>> pred[2:5, 7:13, 9:15] = 1 - pred[2:5, 7:13, 9:15]
        >>> iou = IoU(num_classes=2)
        >>> iou(pred, target)
        tensor(0.9660)

    """

    @deprecated(target=JaccardIndex, deprecated_in="0.7", remove_in="0.8")
    def __init__(
        self,
        num_classes: int,
        ignore_index: Optional[int] = None,
        absent_score: float = 0.0,
        threshold: float = 0.5,
        reduction: str = "elementwise_mean",
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ) -> None:
        void(
            num_classes,
            ignore_index,
            absent_score,
            threshold,
            reduction,
            compute_on_step,
            dist_sync_on_step,
            process_group,
        )
