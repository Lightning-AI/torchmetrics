# Copyright The Lightning team.
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
from torchmetrics.detection.panoptic_qualities import ModifiedPanopticQuality, PanopticQuality
from torchmetrics.utilities.imports import (
    _TORCHVISION_AVAILABLE,
    _TORCHVISION_GREATER_EQUAL_0_8,
    _TORCHVISION_GREATER_EQUAL_0_13,
)

__all__ = ["ModifiedPanopticQuality", "PanopticQuality"]

if _TORCHVISION_GREATER_EQUAL_0_8:
    from torchmetrics.detection.giou import GeneralizedIntersectionOverUnion
    from torchmetrics.detection.iou import IntersectionOverUnion
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    __all__ += ["MeanAveragePrecision", "GeneralizedIntersectionOverUnion", "IntersectionOverUnion"]

if _TORCHVISION_GREATER_EQUAL_0_13:
    from torchmetrics.detection.ciou import CompleteIntersectionOverUnion
    from torchmetrics.detection.diou import DistanceIntersectionOverUnion

    __all__ += ["CompleteIntersectionOverUnion", "DistanceIntersectionOverUnion"]
