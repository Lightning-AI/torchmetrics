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
from functools import partial

import pytest
import torch
from torchmetrics.detection.ciou import CompleteIntersectionOverUnion
from torchmetrics.detection.diou import DistanceIntersectionOverUnion
from torchmetrics.detection.giou import GeneralizedIntersectionOverUnion
from torchmetrics.detection.iou import IntersectionOverUnion
from torchmetrics.functional.detection.ciou import complete_intersection_over_union
from torchmetrics.functional.detection.diou import distance_intersection_over_union
from torchmetrics.functional.detection.giou import generalized_intersection_over_union
from torchmetrics.functional.detection.iou import intersection_over_union
from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_8
from torchvision.ops import box_iou as tv_iou
from torchvision.ops import complete_box_iou as tv_ciou
from torchvision.ops import distance_box_iou as tv_diou
from torchvision.ops import generalized_box_iou as tv_giou

from unittests.helpers.testers import MetricTester

_pytest_condition = not (_TORCHVISION_AVAILABLE and _TORCHVISION_GREATER_EQUAL_0_8)


def _tv_wrapper(preds, target, base_fn, aggregate=True, iou_threshold=None):
    out = base_fn(preds, target)
    if iou_threshold is not None:
        out[out < iou_threshold] = 0
    if aggregate:
        return out.diag().mean()
    return out.diag()


_preds_fn = (
    torch.tensor(
        [
            [296.55, 93.96, 314.97, 152.79],
            [328.94, 97.05, 342.49, 122.98],
            [356.62, 95.47, 372.33, 147.55],
        ]
    )
    .unsqueeze(0)
    .repeat(4, 1, 1)
)
_target_fn = (
    torch.tensor(
        [
            [300.00, 100.00, 315.00, 150.00],
            [330.00, 100.00, 350.00, 125.00],
            [350.00, 100.00, 375.00, 150.00],
        ]
    )
    .unsqueeze(0)
    .repeat(4, 1, 1)
)

_preds_class = [
    [
        {
            "boxes": torch.tensor([[296.55, 93.96, 314.97, 152.79], [298.55, 98.96, 314.97, 151.79]]),
            "labels": torch.tensor([4, 5]),
        }
    ],
    [
        {
            "boxes": torch.tensor([[296.55, 93.96, 314.97, 152.79], [298.55, 98.96, 314.97, 151.79]]),
            "labels": torch.tensor([4, 5]),
        }
    ],
    [
        {
            "boxes": torch.tensor([[328.94, 97.05, 342.49, 122.98]]),
            "scores": torch.tensor([0.456]),
            "labels": torch.tensor([4]),
        },
        {
            "boxes": torch.tensor([[356.62, 95.47, 372.33, 147.55]]),
            "scores": torch.tensor([0.791]),
            "labels": torch.tensor([4]),
        },
    ],
    [
        {
            "boxes": torch.tensor([[328.94, 97.05, 342.49, 122.98]]),
            "scores": torch.tensor([0.456]),
            "labels": torch.tensor([5]),
        },
        {
            "boxes": torch.tensor([[356.62, 95.47, 372.33, 147.55]]),
            "scores": torch.tensor([0.791]),
            "labels": torch.tensor([5]),
        },
    ],
]
target = (
    [
        [
            {
                "boxes": torch.tensor([[300.00, 100.00, 315.00, 150.00]]),
                "labels": torch.tensor([5]),
            }
        ],
        [
            {
                "boxes": torch.tensor([[300.00, 100.00, 315.00, 150.00]]),
                "labels": torch.tensor([5]),
            }
        ],
        [
            {
                "boxes": torch.tensor([[330.00, 100.00, 350.00, 125.00]]),
                "labels": torch.tensor([4]),
            },
            {
                "boxes": torch.tensor([[350.00, 100.00, 375.00, 150.00]]),
                "labels": torch.tensor([4]),
            },
        ],
        [
            {
                "boxes": torch.tensor([[330.00, 100.00, 350.00, 125.00]]),
                "labels": torch.tensor([5]),
            },
            {
                "boxes": torch.tensor([[350.00, 100.00, 375.00, 150.00]]),
                "labels": torch.tensor([4]),
            },
        ],
    ],
)


@pytest.mark.parametrize(
    "class_metric, functional_metric, reference_metric",
    [
        (IntersectionOverUnion, intersection_over_union, tv_iou),
        (CompleteIntersectionOverUnion, complete_intersection_over_union, tv_ciou),
        (DistanceIntersectionOverUnion, distance_intersection_over_union, tv_diou),
        (GeneralizedIntersectionOverUnion, generalized_intersection_over_union, tv_giou),
    ],
)
class TestIntersectionMetrics(MetricTester):
    """Tester class for the different intersection metrics."""

    # @pytest.mark.parametrize("preds, target", [])
    # @pytest.mark.parametrize("ddp", [False, True])
    # def test_intersection_class(self, class_metric, functional_metric, reference_metric, preds, target, ddp):
    #     """Test class implementation for correctness."""
    #     self.run_class_metric_test(
    #         ddp=ddp,
    #         preds=preds,
    #         target=target,
    #         metric_class=class_metric,
    #         reference_metric=reference_metric,
    #     )

    @pytest.mark.parametrize(("preds", "target"), [(_preds_fn, _target_fn)])
    @pytest.mark.parametrize("aggregate", [True, False])
    @pytest.mark.parametrize("iou_threshold", [None, 0.5, 0.7, 0.9])
    def test_intersection_function(
        self, class_metric, functional_metric, reference_metric, preds, target, aggregate, iou_threshold
    ):
        """Test functional implementation for correctness."""
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=functional_metric,
            reference_metric=partial(
                _tv_wrapper, base_fn=reference_metric, aggregate=aggregate, iou_threshold=iou_threshold
            ),
            metric_args={"aggregate": aggregate, "iou_threshold": iou_threshold},
        )
