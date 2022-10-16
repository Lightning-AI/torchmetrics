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
import pytest
import torch

from torchmetrics.detection.box_giou import BoxGeneralizedIntersectionOverUnion
from torchmetrics.detection.giou import GeneralizedIntersectionOverUnion
from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_8
from unittests.detection.test_iou import TestIntersectionOverUnion

giou = torch.Tensor(
    [
        [0.6895, -0.4964, -0.4944],
        [-0.5105, 0.4673, -0.3434],
        [-0.6024, -0.4021, 0.5345],
    ]
)


_pytest_condition = not (_TORCHVISION_AVAILABLE and _TORCHVISION_GREATER_EQUAL_0_8)


@pytest.mark.skipif(_pytest_condition, reason="test requires that torchvision=>0.8.0 is installed")
class TestGeneralizedIntersectionOverUnion(TestIntersectionOverUnion):
    """Test the Generalized Intersection over Union metric for object detection predictions."""

    def __init__(self) -> None:
        super().__init__()
        self.results = giou
        self.metric_class = GeneralizedIntersectionOverUnion
        self.metric_box_class = BoxGeneralizedIntersectionOverUnion
