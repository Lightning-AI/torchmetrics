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

from torchmetrics.functional.detection.panoptic_qualities import modified_panoptic_quality, panoptic_quality
from torchmetrics.utilities.imports import (
    _TORCHVISION_AVAILABLE,
)

__all__ = ["modified_panoptic_quality", "panoptic_quality"]

if _TORCHVISION_AVAILABLE:
    from torchmetrics.functional.detection.ciou import complete_intersection_over_union
    from torchmetrics.functional.detection.diou import distance_intersection_over_union
    from torchmetrics.functional.detection.giou import generalized_intersection_over_union
    from torchmetrics.functional.detection.iou import intersection_over_union

    __all__ += [
        "complete_intersection_over_union",
        "distance_intersection_over_union",
        "generalized_intersection_over_union",
        "intersection_over_union",
    ]
