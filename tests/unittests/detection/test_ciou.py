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
from typing import Callable, ClassVar, Dict

import pytest
import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.detection.ciou import CompleteIntersectionOverUnion
from torchmetrics.functional.detection.ciou import complete_intersection_over_union
from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_13

from unittests.detection.base_iou_test import BaseTestIntersectionOverUnion, TestCaseData, _box_inputs, _inputs
from unittests.helpers.testers import MetricTester

ciou = torch.Tensor(
    [
        [-0.2669985],
    ]
)
ciou_dontrespect = torch.Tensor(
    [
        [0.6078202],
    ]
)
box_ciou = torch.Tensor(
    [
        [0.6883, -0.2072, -0.3352],
        [-0.2217, 0.4881, -0.1913],
        [-0.3971, -0.1543, 0.5606],
    ]
)


_pytest_condition = not (_TORCHVISION_AVAILABLE and _TORCHVISION_GREATER_EQUAL_0_13)


@pytest.mark.skipif(_pytest_condition, reason="test requires that torchvision=>0.13.0 is installed")
class TestCompleteIntersectionOverUnion(MetricTester, BaseTestIntersectionOverUnion):
    """Test the Complete Intersection over Union metric for object detection predictions."""

    data: ClassVar[Dict[str, TestCaseData]] = {
        "iou_variant": TestCaseData(data=_inputs, result={CompleteIntersectionOverUnion._iou_type: ciou}),
        "iou_variant_respect": TestCaseData(
            data=_inputs, result={CompleteIntersectionOverUnion._iou_type: ciou_dontrespect}
        ),
        "fn_iou_variant": TestCaseData(data=_box_inputs, result=box_ciou),
    }
    metric_class: ClassVar[Metric] = CompleteIntersectionOverUnion
    metric_fn: ClassVar[Callable[[Tensor, Tensor, bool, float], Tensor]] = complete_intersection_over_union
