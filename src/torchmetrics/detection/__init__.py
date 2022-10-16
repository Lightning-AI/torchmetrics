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
from torchmetrics.utilities.imports import (
    _TORCHVISION_AVAILABLE,
    _TORCHVISION_GREATER_EQUAL_0_8,
    _TORCHVISION_GREATER_EQUAL_0_13,
)

if _TORCHVISION_AVAILABLE and _TORCHVISION_GREATER_EQUAL_0_13:
    from torchmetrics.detection.box_ciou import BoxCompleteIntersectionOverUnion  # noqa: F401
    from torchmetrics.detection.box_diou import BoxDistanceIntersectionOverUnion  # noqa: F401
    from torchmetrics.detection.ciou import CompleteIntersectionOverUnion  # noqa: F401
    from torchmetrics.detection.diou import DistanceIntersectionOverUnion  # noqa: F401

if _TORCHVISION_AVAILABLE and _TORCHVISION_GREATER_EQUAL_0_8:
    from torchmetrics.detection.box_giou import BoxGeneralizedIntersectionOverUnion  # noqa: F401
    from torchmetrics.detection.box_iou import BoxIntersectionOverUnion  # noqa: F401
    from torchmetrics.detection.giou import GeneralizedIntersectionOverUnion  # noqa: F401
    from torchmetrics.detection.iou import IntersectionOverUnion  # noqa: F401
    from torchmetrics.detection.mean_ap import MeanAveragePrecision  # noqa: F401
