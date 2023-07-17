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
from torchmetrics.detection.giou import GeneralizedIntersectionOverUnion
from torchmetrics.functional.detection.giou import generalized_intersection_over_union
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_8

from unittests.detection.base_iou_test import BaseTestIntersectionOverUnion, TestCaseData, _box_inputs, _inputs
from unittests.helpers.testers import MetricTester

giou = torch.Tensor(
    [
        [0.05507809],
    ]
)
giou_dontrespect = torch.Tensor(
    [
        [0.59242314],
    ]
)
box_giou = torch.Tensor(
    [
        [0.6895, -0.4964, -0.4944],
        [-0.5105, 0.4673, -0.3434],
        [-0.6024, -0.4021, 0.5345],
    ]
)

_pytest_condition = not (_TORCHVISION_AVAILABLE and _TORCHVISION_GREATER_EQUAL_0_8)


@pytest.mark.skipif(_pytest_condition, reason="test requires that torchvision=>0.8.0 is installed")
class TestGeneralizedIntersectionOverUnion(MetricTester, BaseTestIntersectionOverUnion):
    """Test the Generalized Intersection over Union metric for object detection
    predictions.
    """

    data: ClassVar[Dict[str, TestCaseData]] = {
        "iou_variant": TestCaseData(data=_inputs, result={GeneralizedIntersectionOverUnion._iou_type: giou}),
        "iou_variant_respect": TestCaseData(
            data=_inputs, result={GeneralizedIntersectionOverUnion._iou_type: giou_dontrespect}
        ),
        "fn_iou_variant": TestCaseData(data=_box_inputs, result=box_giou),
    }
    metric_class: ClassVar[Metric] = GeneralizedIntersectionOverUnion
    metric_fn: ClassVar[Callable[[Tensor, Tensor, bool, float], Tensor]] = generalized_intersection_over_union
