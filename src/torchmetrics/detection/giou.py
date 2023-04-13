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
from typing import Any, Callable, Optional

from torch import Tensor

from torchmetrics.detection.iou import IntersectionOverUnion
from torchmetrics.functional.detection.giou import _giou_compute, _giou_update


class GeneralizedIntersectionOverUnion(IntersectionOverUnion):
    r"""Compute Generalized Intersection Over Union (GIoU) <https://arxiv.org/abs/1902.09630>`_.

    Args:
        box_format:
            Input format of given boxes. Supported formats are ``[`xyxy`, `xywh`, `cxcywh`]``.
        iou_thresholds:
            Optional IoU thresholds for evaluation. If set to `None` the threshold is ignored.
        class_metrics:
            Option to enable per-class metrics for IoU. Has a performance impact.
        kwargs:
            Additional keyword arguments, see :ref:`Metric kwargs` for more info.
    """
    _iou_type: str = "giou"
    _invalid_val: float = -1.0

    def __init__(
        self,
        box_format: str = "xyxy",
        iou_threshold: Optional[float] = None,
        class_metrics: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(box_format, iou_threshold, class_metrics, **kwargs)

    @staticmethod
    def _iou_update_fn(*args: Any, **kwargs: Any) -> Tensor:
        return _giou_update(*args, **kwargs)

    @staticmethod
    def _iou_compute_fn(*args: Any, **kwargs: Any) -> Tensor:
        return _giou_compute(*args, **kwargs)
