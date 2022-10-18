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
from typing import Dict

import torch

from torchmetrics.detection.giou import GeneralizedIntersectionOverUnion
from unittests.detection.base_iou_test import BaseTestIntersectionOverUnion, TestCaseData, _box_inputs, _inputs
from unittests.helpers.testers import MetricTester

giou = torch.Tensor(
    [
        [0.563799],
    ]
)
box_giou = torch.Tensor(
    [
        [0.6895, -0.4964, -0.4944],
        [-0.5105, 0.4673, -0.3434],
        [-0.6024, -0.4021, 0.5345],
    ]
)


class TestGeneralizedIntersectionOverUnion(MetricTester, BaseTestIntersectionOverUnion):
    """Test the Generalized Intersection over Union metric for object detection predictions."""

    data: Dict[str, TestCaseData] = {
        "iou_variant": TestCaseData(data=_inputs, result={GeneralizedIntersectionOverUnion.type: giou}),
        "box_iou_variant": TestCaseData(data=_box_inputs, result=box_giou),
    }
    metric_class = GeneralizedIntersectionOverUnion
