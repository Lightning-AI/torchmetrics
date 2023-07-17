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
from abc import ABC
from collections import namedtuple
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ClassVar, Dict

import pytest
import torch
from torch import IntTensor, Tensor
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_8

Input = namedtuple("Input", ["preds", "target"])


@dataclass
class TestCaseData:
    """Test data sample."""

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


def compare_fn(preds: Any, target: Any, result: Any):
    """Mock compare function by returning additional parameter results
    directly.
    """
    return result


@pytest.mark.skipif(_pytest_condition, reason="test requires that torchvision=>0.8.0 is installed")
@pytest.mark.parametrize("compute_on_cpu", [True, False])
@pytest.mark.parametrize("ddp", [False, True])
class BaseTestIntersectionOverUnion(ABC):
    """Base Test the Intersection over Union metric for object detection
    predictions.
    """

    data: ClassVar[Dict[str, TestCaseData]] = {
        "iou_variant": TestCaseData(data=_inputs, result={"iou": torch.Tensor([0])}),
        "fn_iou_variant": TestCaseData(data=_box_inputs, result=None),
    }
    metric_class: ClassVar
    metric_fn: Callable[[Tensor, Tensor, bool, float], Tensor]

    def test_iou_variant(self, compute_on_cpu: bool, ddp: bool):
        """Test modular implementation for correctness."""
        key = "iou_variant"

        self.run_class_metric_test(  # type: ignore
            ddp=ddp,
            preds=self.data[key].data.preds,
            target=self.data[key].data.target,
            metric_class=self.metric_class,
            reference_metric=partial(compare_fn, result=self.data[key].result),
            dist_sync_on_step=False,
            check_batch=False,
            metric_args={"compute_on_cpu": compute_on_cpu},
        )

    def test_iou_variant_dont_respect_labels(self, compute_on_cpu: bool, ddp: bool):
        """Test modular implementation for correctness while ignoring
        labels.
        """
        key = "iou_variant_respect"

        self.run_class_metric_test(  # type: ignore
            ddp=ddp,
            preds=self.data[key].data.preds,
            target=self.data[key].data.target,
            metric_class=self.metric_class,
            reference_metric=partial(compare_fn, result=self.data[key].result),
            dist_sync_on_step=False,
            check_batch=False,
            metric_args={"compute_on_cpu": compute_on_cpu, "respect_labels": False},
        )

    def test_fn(self, compute_on_cpu: bool, ddp: bool):
        """Test functional implementation for correctness."""
        key = "fn_iou_variant"
        self.run_functional_metric_test(
            self.data[key].data.preds[0].unsqueeze(0),  # pass as batch, otherwise it attempts to pass element wise
            self.data[key].data.target[0].unsqueeze(0),
            self.metric_fn.__func__,
            partial(compare_fn, result=self.data[key].result),
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
