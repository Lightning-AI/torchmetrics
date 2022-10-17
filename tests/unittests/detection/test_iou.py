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

from torchmetrics.detection.iou import IntersectionOverUnion
from unittests.detection.base_iou_test import BaseTestIntersectionOverUnion, TestCaseData, _box_inputs, _inputs
from unittests.helpers.testers import MetricTester

iou = torch.Tensor(
    [
        [0.58793],
    ]
)
box_iou = torch.Tensor(
    [
        [0.6898, 0.0000, 0.0000],
        [0.0000, 0.5086, 0.0000],
        [0.0000, 0.0000, 0.5654],
    ]
)


class TestIntersectionOverUnion(MetricTester, BaseTestIntersectionOverUnion):
    """Test the Intersection over Union metric for object detection predictions."""

    data: Dict[str, TestCaseData] = {
        "iou_variant": TestCaseData(data=_inputs, result={IntersectionOverUnion.type: iou}),
        "box_iou_variant": TestCaseData(data=_box_inputs, result=box_iou),
    }
    metric_class = IntersectionOverUnion
