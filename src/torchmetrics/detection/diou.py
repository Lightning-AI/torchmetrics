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
from torchmetrics.functional.detection.diou import _diou_compute, _diou_update
from torchmetrics.utilities.imports import _TORCHVISION_GREATER_EQUAL_0_8, _TORCHVISION_GREATER_EQUAL_0_13

if _TORCHVISION_GREATER_EQUAL_0_8:
    from torchvision.ops import box_convert
else:
    box_convert = None
    __doctest_skip__ = ["DistanceIntersectionOverUnion"]


class DistanceIntersectionOverUnion(IntersectionOverUnion):
    r"""
    Computes Distance Intersection Over Union (DIoU) <https://arxiv.org/abs/1911.08287v1>`_
    Args:
        box_format:
            Input format of given boxes. Supported formats are ``[`xyxy`, `xywh`, `cxcywh`]``.
        iou_thresholds:
            Optional IoU thresholds for evaluation. If set to `None` the threshold is ignored.
        class_metrics:
            Option to enable per-class metrics for IoU. Has a performance impact.
    """

    update_fn: Callable[[Tensor, Tensor, bool], Tensor] = _diou_update
    compute_fn: Callable[[Tensor], Tensor] = _diou_compute
    type: str = "diou"

    def __init__(
        self,
        box_format: str = "xyxy",
        iou_threshold: Optional[float] = None,
        class_metrics: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(box_format, iou_threshold, class_metrics, **kwargs)

        if not _TORCHVISION_GREATER_EQUAL_0_13:
            raise ModuleNotFoundError(
                f"Metric `{self.type.upper()}` requires that `torchvision` version 0.13.0 or newer is installed."
                " Please install with `pip install torchvision>=0.13` or `pip install torchmetrics[detection]`."
            )
