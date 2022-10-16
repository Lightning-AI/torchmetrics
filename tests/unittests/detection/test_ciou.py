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

from torchmetrics.detection.box_ciou import BoxCompleteIntersectionOverUnion
from torchmetrics.detection.ciou import CompleteIntersectionOverUnion
from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_13
from unittests.detection.test_iou import TestIntersectionOverUnion

ciou = torch.Tensor(
    [
        [0.6883, -0.2072, -0.3352],
        [-0.2217, 0.4881, -0.1913],
        [-0.3971, -0.1543, 0.5606],
    ]
)


_pytest_condition = not (_TORCHVISION_AVAILABLE and _TORCHVISION_GREATER_EQUAL_0_13)


@pytest.mark.skipif(_pytest_condition, reason="test requires that torchvision=>0.13.0 is installed")
class TestCompleteIntersectionOverUnion(TestIntersectionOverUnion):
    """Test the Complete Intersection over Union metric for object detection predictions."""

    def __init__(self) -> None:
        super().__init__()
        self.results = ciou
        self.metric_class = CompleteIntersectionOverUnion
        self.metric_box_class = BoxCompleteIntersectionOverUnion
