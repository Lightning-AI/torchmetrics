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
from dataclasses import dataclass
from typing import Any, Dict

import pytest
import torch
from torch import IntTensor, Tensor

from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_8

Input = namedtuple("Input", ["preds", "target"])


@dataclass
class TestCaseData:
    data: Input
    result: Any


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
            dict(
                boxes=Tensor([[296.55, 93.96, 314.97, 152.79]]),
                scores=Tensor([0.236]),
                labels=IntTensor([4]),
            )
        ],
        [
            dict(
                boxes=Tensor([[328.94, 97.05, 342.49, 122.98]]),
                scores=Tensor([0.456]),
                labels=IntTensor([4]),
            ),
            dict(
                boxes=Tensor([[356.62, 95.47, 372.33, 147.55]]),
                scores=Tensor([0.791]),
                labels=IntTensor([4]),
            ),
        ],
    ],
    target=[
        [
            dict(
                boxes=Tensor([[300.00, 100.00, 315.00, 150.00]]),
                labels=IntTensor([4]),
            )
        ],
        [
            dict(
                boxes=Tensor([[330.00, 100.00, 350.00, 125.00]]),
                labels=IntTensor([4]),
            ),
            dict(
                boxes=Tensor([[350.00, 100.00, 375.00, 150.00]]),
                labels=IntTensor([4]),
            ),
        ],
    ],
)

_box_inputs = Input(preds=_preds, target=_target)

_pytest_condition = not (_TORCHVISION_AVAILABLE and _TORCHVISION_GREATER_EQUAL_0_8)


@pytest.mark.skipif(_pytest_condition, reason="test requires that torchvision=>0.8.0 is installed")
@pytest.mark.parametrize("compute_on_cpu", [True, False])
@pytest.mark.parametrize("ddp", [False, True])
class BaseTestIntersectionOverUnion:
    """Base Test the Intersection over Union metric for object detection predictions."""

    data: Dict[str, TestCaseData] = {
        "iou_variant": TestCaseData(data=_inputs, result=None),
        "box_iou_variant": TestCaseData(data=_box_inputs, result=None),
    }
    metric_class: Metric = None

    def test_iou_variant(self, compute_on_cpu: bool, ddp: bool):
        """Test modular implementation for correctness."""
        key = "iou_variant"
        self.run_class_metric_test(
            ddp=ddp,
            preds=self.data[key].data.preds,
            target=self.data[key].data.target,
            metric_class=self.metric_class,
            sk_metric=self.data[key].result,
            dist_sync_on_step=False,
            check_batch=False,
            metric_args={"compute_on_cpu": compute_on_cpu},
        )

    def test_error_on_wrong_input(self, compute_on_cpu: bool, ddp: bool):
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
