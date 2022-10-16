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

import pytest
import torch
from torch import IntTensor, Tensor

from torchmetrics.detection.box_iou import BoxIntersectionOverUnion
from torchmetrics.detection.iou import IntersectionOverUnion
from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_8
from unittests.helpers.testers import MetricTester

Input = namedtuple("Input", ["preds", "target"])

preds = torch.Tensor(
    [
        [296.55, 93.96, 314.97, 152.79],
        [328.94, 97.05, 342.49, 122.98],
        [356.62, 95.47, 372.33, 147.55],
    ]
)
target = torch.Tensor(
    [
        [300.00, 100.00, 315.00, 150.00],
        [330.00, 100.00, 350.00, 125.00],
        [350.00, 100.00, 375.00, 150.00],
    ]
)
iou = torch.Tensor(
    [
        [0.6898, 0.0000, 0.0000],
        [0.0000, 0.5086, 0.0000],
        [0.0000, 0.0000, 0.5654],
    ]
)


_pytest_condition = not (_TORCHVISION_AVAILABLE and _TORCHVISION_GREATER_EQUAL_0_8)


@pytest.mark.skipif(_pytest_condition, reason="test requires that torchvision=>0.8.0 is installed")
@pytest.mark.parametrize("compute_on_cpu", [True, False])
@pytest.mark.parametrize("ddp", [False, True])
class TestIntersectionOverUnion(MetricTester):
    """Test the Intersection over Union metric for object detection predictions."""

    def __init__(self) -> None:
        super().__init__()
        self.results = iou
        self.metric_class = IntersectionOverUnion
        self.metric_box_class = BoxIntersectionOverUnion

    def test_iou_variant(self, compute_on_cpu, ddp):
        """Test modular implementation for correctness."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=self.metric_class,
            sk_metric=self.results,
            dist_sync_on_step=False,
            check_batch=False,
            metric_args={"compute_on_cpu": compute_on_cpu},
        )

    def test_box_iou_variant(self, compute_on_cpu, ddp):
        """Test modular implementation for correctness."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=self.metric_box_class,
            sk_metric=self.results,
            dist_sync_on_step=False,
            check_batch=False,
            metric_args={"compute_on_cpu": compute_on_cpu},
        )

    def test_error_on_wrong_input(self):
        """Test class input validation."""
        metric = self.metric_class()

        metric.update([], [])  # no error

        with pytest.raises(ValueError, match="Expected argument `preds` to be of type Sequence"):
            metric.update(Tensor(), [])  # type: ignore

        with pytest.raises(ValueError, match="Expected argument `target` to be of type Sequence"):
            metric.update([], Tensor())  # type: ignore

        with pytest.raises(ValueError, match="Expected argument `preds` and `target` to have the same length"):
            metric.update([{}], [{}, {}])

        with pytest.raises(ValueError, match="Expected all dicts in `preds` to contain the `boxes` key"):
            metric.update(
                [dict(scores=Tensor(), labels=IntTensor)],
                [dict(boxes=Tensor(), labels=IntTensor())],
            )

        with pytest.raises(ValueError, match="Expected all dicts in `preds` to contain the `scores` key"):
            metric.update(
                [dict(boxes=Tensor(), labels=IntTensor)],
                [dict(boxes=Tensor(), labels=IntTensor())],
            )

        with pytest.raises(ValueError, match="Expected all dicts in `preds` to contain the `labels` key"):
            metric.update(
                [dict(boxes=Tensor(), scores=IntTensor)],
                [dict(boxes=Tensor(), labels=IntTensor())],
            )

        with pytest.raises(ValueError, match="Expected all dicts in `target` to contain the `boxes` key"):
            metric.update(
                [dict(boxes=Tensor(), scores=IntTensor, labels=IntTensor)],
                [dict(labels=IntTensor())],
            )

        with pytest.raises(ValueError, match="Expected all dicts in `target` to contain the `labels` key"):
            metric.update(
                [dict(boxes=Tensor(), scores=IntTensor, labels=IntTensor)],
                [dict(boxes=IntTensor())],
            )

        with pytest.raises(ValueError, match="Expected all boxes in `preds` to be of type Tensor"):
            metric.update(
                [dict(boxes=[], scores=Tensor(), labels=IntTensor())],
                [dict(boxes=Tensor(), labels=IntTensor())],
            )

        with pytest.raises(ValueError, match="Expected all scores in `preds` to be of type Tensor"):
            metric.update(
                [dict(boxes=Tensor(), scores=[], labels=IntTensor())],
                [dict(boxes=Tensor(), labels=IntTensor())],
            )

        with pytest.raises(ValueError, match="Expected all labels in `preds` to be of type Tensor"):
            metric.update(
                [dict(boxes=Tensor(), scores=Tensor(), labels=[])],
                [dict(boxes=Tensor(), labels=IntTensor())],
            )

        with pytest.raises(ValueError, match="Expected all boxes in `target` to be of type Tensor"):
            metric.update(
                [dict(boxes=Tensor(), scores=Tensor(), labels=IntTensor())],
                [dict(boxes=[], labels=IntTensor())],
            )

        with pytest.raises(ValueError, match="Expected all labels in `target` to be of type Tensor"):
            metric.update(
                [dict(boxes=Tensor(), scores=Tensor(), labels=IntTensor())],
                [dict(boxes=Tensor(), labels=[])],
            )
