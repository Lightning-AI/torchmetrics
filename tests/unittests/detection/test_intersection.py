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
from collections import namedtuple
from functools import partial
from typing import Any, Callable, ClassVar, Dict

import pytest
import torch
from torch import IntTensor, Tensor
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_8, _TORCHVISION_GREATER_EQUAL_0_13

from unittests.helpers.testers import MetricTester

if _TORCHVISION_AVAILABLE and _TORCHVISION_GREATER_EQUAL_0_13:
    from torchvision.ops import box_iou as tv_iou
    from torchvision.ops import generalized_box_iou as tv_giou
    from torchvision.ops import complete_box_iou as tv_ciou
    from torchvision.ops import distance_box_iou as tv_diou


Input = namedtuple("Input", ["preds", "target"])


_preds = torch.Tensor(
    [
        [296.55, 93.96, 314.97, 152.79],
        [328.94, 97.05, 342.49, 122.98],
        [356.62, 95.47, 372.33, 147.55],
    ]
)
_target = torch.Tensor(
    [
        [300.00, 100.00, 315.00, 150.00],
        [330.00, 100.00, 350.00, 125.00],
        [350.00, 100.00, 375.00, 150.00],
    ]
)

_inputs = Input(
    preds=[
        [
            {
                "boxes": Tensor([[296.55, 93.96, 314.97, 152.79], [298.55, 98.96, 314.97, 151.79]]),
                "scores": Tensor([0.236, 0.56]),
                "labels": IntTensor([4, 5]),
            }
        ],
        [
            {
                "boxes": Tensor([[296.55, 93.96, 314.97, 152.79], [298.55, 98.96, 314.97, 151.79]]),
                "scores": Tensor([0.236, 0.56]),
                "labels": IntTensor([4, 5]),
            }
        ],
        [
            {
                "boxes": Tensor([[328.94, 97.05, 342.49, 122.98]]),
                "scores": Tensor([0.456]),
                "labels": IntTensor([4]),
            },
            {
                "boxes": Tensor([[356.62, 95.47, 372.33, 147.55]]),
                "scores": Tensor([0.791]),
                "labels": IntTensor([4]),
            },
        ],
        [
            {
                "boxes": Tensor([[328.94, 97.05, 342.49, 122.98]]),
                "scores": Tensor([0.456]),
                "labels": IntTensor([5]),
            },
            {
                "boxes": Tensor([[356.62, 95.47, 372.33, 147.55]]),
                "scores": Tensor([0.791]),
                "labels": IntTensor([5]),
            },
        ],
    ],
    target=[
        [
            {
                "boxes": Tensor([[300.00, 100.00, 315.00, 150.00]]),
                "labels": IntTensor([5]),
            }
        ],
        [
            {
                "boxes": Tensor([[300.00, 100.00, 315.00, 150.00]]),
                "labels": IntTensor([5]),
            }
        ],
        [
            {
                "boxes": Tensor([[330.00, 100.00, 350.00, 125.00]]),
                "labels": IntTensor([4]),
            },
            {
                "boxes": Tensor([[350.00, 100.00, 375.00, 150.00]]),
                "labels": IntTensor([4]),
            },
        ],
        [
            {
                "boxes": Tensor([[330.00, 100.00, 350.00, 125.00]]),
                "labels": IntTensor([5]),
            },
            {
                "boxes": Tensor([[350.00, 100.00, 375.00, 150.00]]),
                "labels": IntTensor([4]),
            },
        ],
    ],
)
_box_inputs = Input(preds=_preds, target=_target)

_pytest_condition = not (_TORCHVISION_AVAILABLE and _TORCHVISION_GREATER_EQUAL_0_8)


from torchmetrics.functional.detection.iou import intersection_over_union
from torchmetrics.functional.detection.ciou import complete_intersection_over_union
from torchmetrics.functional.detection.diou import distance_intersection_over_union
from torchmetrics.functional.detection.giou import generalized_intersection_over_union

from torchmetrics.detection.iou import IntersectionOverUnion
from torchmetrics.detection.ciou import CompleteIntersectionOverUnion
from torchmetrics.detection.diou import DistanceIntersectionOverUnion
from torchmetrics.detection.giou import GeneralizedIntersectionOverUnion


@pytest.mark.parametrize("class_metric, functional_metric, reference_metric",
    [
        (IntersectionOverUnion, intersection_over_union, tv_iou),
        (CompleteIntersectionOverUnion, complete_intersection_over_union, tv_ciou),
        (DistanceIntersectionOverUnion, distance_intersection_over_union, tv_diou),
        (GeneralizedIntersectionOverUnion, generalized_intersection_over_union, tv_giou),
    ]
)

class TestIntersectionMetrics(MetricTester):
    @pytest.mark.parametrize("preds, target", [
        (_inputs.preds, _inputs.target),
    ])
    @pytest.mark.parametrize("ddp", [False, True])
    def test_intersection_class(self, class_metric, functional_metric, reference_metric, preds, target, ddp):
        """Test class implementation for correctness."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=class_metric,
            reference_metric=reference_metric,
        )

    @pytest.mark.parametrize("preds, target", [
        #(_box_inputs.preds, _box_inputs.target),
        (torch.)
    ])
    def test_intersection_function(self, class_metric, functional_metric, reference_metric, preds, target):
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=functional_metric,
            reference_metric=reference_metric,
        )


    def test_error_on_wrong_input(self, class_metric, functional_metric, reference_metric):
        """Test class input validation."""
        metric = class_metric()

        metric.update([], [])  # no error

        with pytest.raises(ValueError, match="Expected argument `preds` to be of type Sequence"):
            metric.update(Tensor(), [])  # type: ignore

        with pytest.raises(ValueError, match="Expected argument `target` to be of type Sequence"):
            metric.update([], Tensor())  # type: ignore

        with pytest.raises(ValueError, match="Expected argument `preds` and `target` to have the same length"):
            metric.update([{}], [{}, {}])

        with pytest.raises(ValueError, match="Expected all dicts in `preds` to contain the `boxes` key"):
            metric.update(
                [{"scores": Tensor(), "labels": IntTensor}],
                [{"boxes": Tensor(), "labels": IntTensor()}],
            )

        with pytest.raises(ValueError, match="Expected all dicts in `preds` to contain the `scores` key"):
            metric.update(
                [{"boxes": Tensor(), "labels": IntTensor}],
                [{"boxes": Tensor(), "labels": IntTensor()}],
            )

        with pytest.raises(ValueError, match="Expected all dicts in `preds` to contain the `labels` key"):
            metric.update(
                [{"boxes": Tensor(), "scores": IntTensor}],
                [{"boxes": Tensor(), "labels": IntTensor()}],
            )

        with pytest.raises(ValueError, match="Expected all dicts in `target` to contain the `boxes` key"):
            metric.update(
                [{"boxes": Tensor(), "scores": IntTensor, "labels": IntTensor}],
                [{"labels": IntTensor()}],
            )

        with pytest.raises(ValueError, match="Expected all dicts in `target` to contain the `labels` key"):
            metric.update(
                [{"boxes": Tensor(), "scores": IntTensor, "labels": IntTensor}],
                [{"boxes": IntTensor()}],
            )

        with pytest.raises(ValueError, match="Expected all boxes in `preds` to be of type Tensor"):
            metric.update(
                [{"boxes": [], "scores": Tensor(), "labels": IntTensor()}],
                [{"boxes": Tensor(), "labels": IntTensor()}],
            )

        with pytest.raises(ValueError, match="Expected all scores in `preds` to be of type Tensor"):
            metric.update(
                [{"boxes": Tensor(), "scores": [], "labels": IntTensor()}],
                [{"boxes": Tensor(), "labels": IntTensor()}],
            )

        with pytest.raises(ValueError, match="Expected all labels in `preds` to be of type Tensor"):
            metric.update(
                [{"boxes": Tensor(), "scores": Tensor(), "labels": []}],
                [{"boxes": Tensor(), "labels": IntTensor()}],
            )

        with pytest.raises(ValueError, match="Expected all boxes in `target` to be of type Tensor"):
            metric.update(
                [{"boxes": Tensor(), "scores": Tensor(), "labels": IntTensor()}],
                [{"boxes": [], "labels": IntTensor()}],
            )

        with pytest.raises(ValueError, match="Expected all labels in `target` to be of type Tensor"):
            metric.update(
                [{"boxes": Tensor(), "scores": Tensor(), "labels": IntTensor()}],
                [{"boxes": Tensor(), "labels": []}],
            )
