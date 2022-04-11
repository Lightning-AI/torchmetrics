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

from tests.helpers.testers import MetricTester
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_8

Input = namedtuple("Input", ["preds", "target"])

_inputs = Input(
    preds=[
        [
            dict(
                boxes=torch.Tensor([[258.15, 41.29, 606.41, 285.07]]),
                scores=torch.Tensor([0.236]),
                labels=torch.IntTensor([4]),
            ),  # coco image id 42
            dict(
                boxes=torch.Tensor([[61.00, 22.75, 565.00, 632.42], [12.66, 3.32, 281.26, 275.23]]),
                scores=torch.Tensor([0.318, 0.726]),
                labels=torch.IntTensor([3, 2]),
            ),  # coco image id 73
        ],
        [
            dict(
                boxes=torch.Tensor(
                    [
                        [87.87, 276.25, 384.29, 379.43],
                        [0.00, 3.66, 142.15, 316.06],
                        [296.55, 93.96, 314.97, 152.79],
                        [328.94, 97.05, 342.49, 122.98],
                        [356.62, 95.47, 372.33, 147.55],
                        [464.08, 105.09, 495.74, 146.99],
                        [276.11, 103.84, 291.44, 150.72],
                    ]
                ),
                scores=torch.Tensor([0.546, 0.3, 0.407, 0.611, 0.335, 0.805, 0.953]),
                labels=torch.IntTensor([4, 1, 0, 0, 0, 0, 0]),
            ),  # coco image id 74
            dict(
                boxes=torch.Tensor([[0.00, 2.87, 601.00, 421.52]]),
                scores=torch.Tensor([0.699]),
                labels=torch.IntTensor([5]),
            ),  # coco image id 133
        ],
    ],
    target=[
        [
            dict(
                boxes=torch.Tensor([[214.1500, 41.2900, 562.4100, 285.0700]]),
                labels=torch.IntTensor([4]),
            ),  # coco image id 42
            dict(
                boxes=torch.Tensor(
                    [
                        [13.00, 22.75, 548.98, 632.42],
                        [1.66, 3.32, 270.26, 275.23],
                    ]
                ),
                labels=torch.IntTensor([2, 2]),
            ),  # coco image id 73
        ],
        [
            dict(
                boxes=torch.Tensor(
                    [
                        [61.87, 276.25, 358.29, 379.43],
                        [2.75, 3.66, 162.15, 316.06],
                        [295.55, 93.96, 313.97, 152.79],
                        [326.94, 97.05, 340.49, 122.98],
                        [356.62, 95.47, 372.33, 147.55],
                        [462.08, 105.09, 493.74, 146.99],
                        [277.11, 103.84, 292.44, 150.72],
                    ]
                ),
                labels=torch.IntTensor([4, 1, 0, 0, 0, 0, 0]),
            ),  # coco image id 74
            dict(
                boxes=torch.Tensor([[13.99, 2.87, 640.00, 421.52]]),
                labels=torch.IntTensor([5]),
            ),  # coco image id 133
        ],
    ],
)

# example from this issue https://github.com/PyTorchLightning/metrics/issues/943
_inputs2 = Input(
    preds=[
        [
            dict(
                boxes=torch.Tensor([[258.0, 41.0, 606.0, 285.0]]),
                scores=torch.Tensor([0.536]),
                labels=torch.IntTensor([0]),
            ),
        ],
        [
            dict(
                boxes=torch.Tensor([[258.0, 41.0, 606.0, 285.0]]),
                scores=torch.Tensor([0.536]),
                labels=torch.IntTensor([0]),
            )
        ],
    ],
    target=[
        [
            dict(
                boxes=torch.Tensor([[214.0, 41.0, 562.0, 285.0]]),
                labels=torch.IntTensor([0]),
            )
        ],
        [
            dict(
                boxes=torch.Tensor([]),
                labels=torch.IntTensor([]),
            )
        ],
    ],
)


def _compare_fn(preds, target) -> dict:
    """Comparison function for map implementation.

    Official pycocotools results calculated from a subset of https://github.com/cocodataset/cocoapi/tree/master/results
        All classes
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.706
        Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.901
        Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.846
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.689
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.800
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.701
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.592
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.716
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.716
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.767
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.800
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.700

        Class 0
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.725
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.780

        Class 1
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.800
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.800

        Class 2
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.454
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.450

        Class 3
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = -1.000
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = -1.000

        Class 4
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.650
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.650

        Class 5
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.900
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.900
    """
    return {
        "map": torch.Tensor([0.706]),
        "map_50": torch.Tensor([0.901]),
        "map_75": torch.Tensor([0.846]),
        "map_small": torch.Tensor([0.689]),
        "map_medium": torch.Tensor([0.800]),
        "map_large": torch.Tensor([0.701]),
        "mar_1": torch.Tensor([0.592]),
        "mar_10": torch.Tensor([0.716]),
        "mar_100": torch.Tensor([0.716]),
        "mar_small": torch.Tensor([0.767]),
        "mar_medium": torch.Tensor([0.800]),
        "mar_large": torch.Tensor([0.700]),
        "map_per_class": torch.Tensor([0.725, 0.800, 0.454, -1.000, 0.650, 0.900]),
        "mar_100_per_class": torch.Tensor([0.780, 0.800, 0.450, -1.000, 0.650, 0.900]),
    }


_pytest_condition = not (_TORCHVISION_AVAILABLE and _TORCHVISION_GREATER_EQUAL_0_8)


@pytest.mark.skipif(_pytest_condition, reason="test requires that torchvision=>0.8.0 is installed")
@pytest.mark.parametrize("compute_on_cpu", [True, False])
class TestMAP(MetricTester):
    """Test the MAP metric for object detection predictions.

    Results are compared to original values from the pycocotools implementation.
    A subset of the first 10 fake predictions of the official repo is used:
    https://github.com/cocodataset/cocoapi/blob/master/results/instances_val2014_fakebbox100_results.json
    """

    atol = 1e-1

    @pytest.mark.parametrize("ddp", [False, True])
    def test_map(self, compute_on_cpu, ddp):
        """Test modular implementation for correctness."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=_inputs.preds,
            target=_inputs.target,
            metric_class=MeanAveragePrecision,
            sk_metric=_compare_fn,
            dist_sync_on_step=False,
            check_batch=False,
            metric_args={"class_metrics": True, "compute_on_cpu": compute_on_cpu},
        )


# noinspection PyTypeChecker
@pytest.mark.skipif(_pytest_condition, reason="test requires that torchvision=>0.8.0 is installed")
def test_error_on_wrong_init():
    """Test class raises the expected errors."""
    MeanAveragePrecision()  # no error

    with pytest.raises(ValueError, match="Expected argument `class_metrics` to be a boolean"):
        MeanAveragePrecision(class_metrics=0)


@pytest.mark.skipif(_pytest_condition, reason="test requires that torchvision=>0.8.0 is installed")
def test_empty_preds():
    """Test empty predictions."""
    metric = MeanAveragePrecision()

    metric.update(
        [
            dict(boxes=torch.Tensor([]), scores=torch.Tensor([]), labels=torch.IntTensor([])),
        ],
        [
            dict(boxes=torch.Tensor([[214.1500, 41.2900, 562.4100, 285.0700]]), labels=torch.IntTensor([4])),
        ],
    )
    metric.compute()


@pytest.mark.skipif(_pytest_condition, reason="test requires that torchvision=>0.8.0 is installed")
def test_empty_ground_truths():
    """Test empty ground truths."""
    metric = MeanAveragePrecision()

    metric.update(
        [
            dict(
                boxes=torch.Tensor([[214.1500, 41.2900, 562.4100, 285.0700]]),
                scores=torch.Tensor([0.5]),
                labels=torch.IntTensor([4]),
            ),
        ],
        [
            dict(boxes=torch.Tensor([]), labels=torch.IntTensor([])),
        ],
    )
    metric.compute()


_gpu_test_condition = not torch.cuda.is_available()


def _move_to_gpu(input):
    for x in input:
        for key in x.keys():
            if torch.is_tensor(x[key]):
                x[key] = x[key].to("cuda")
    return input


@pytest.mark.skipif(_pytest_condition, reason="test requires that torchvision=>0.8.0 is installed")
@pytest.mark.skipif(_gpu_test_condition, reason="test requires CUDA availability")
@pytest.mark.parametrize("inputs", [_inputs, _inputs2])
def test_map_gpu(inputs):
    """Test predictions on single gpu."""
    metric = MeanAveragePrecision()
    metric = metric.to("cuda")
    for preds, targets in zip(inputs.preds, inputs.target):
        metric.update(_move_to_gpu(preds), _move_to_gpu(targets))
    metric.compute()


@pytest.mark.skipif(_pytest_condition, reason="test requires that pycocotools and torchvision=>0.8.0 is installed")
def test_empty_metric():
    """Test empty metric."""
    metric = MeanAveragePrecision()
    metric.compute()


@pytest.mark.skipif(_pytest_condition, reason="test requires that pycocotools and torchvision=>0.8.0 is installed")
def test_missing_pred():
    """One good detection, one false negative.

    Map should be lower than 1. Actually it is 0.5, but the exact value depends on where we are sampling (i.e. recall's
    values)
    """
    gts = [
        dict(boxes=torch.Tensor([[10, 20, 15, 25]]), labels=torch.IntTensor([0])),
        dict(boxes=torch.Tensor([[10, 20, 15, 25]]), labels=torch.IntTensor([0])),
    ]
    preds = [
        dict(boxes=torch.Tensor([[10, 20, 15, 25]]), scores=torch.Tensor([0.9]), labels=torch.IntTensor([0])),
        # Empty prediction
        dict(boxes=torch.Tensor([]), scores=torch.Tensor([]), labels=torch.IntTensor([])),
    ]
    metric = MeanAveragePrecision()
    metric.update(preds, gts)
    result = metric.compute()
    assert result["map"] < 1, "MAP cannot be 1, as there is a missing prediction."


@pytest.mark.skipif(_pytest_condition, reason="test requires that pycocotools and torchvision=>0.8.0 is installed")
def test_missing_gt():
    """The symmetric case of test_missing_pred.

    One good detection, one false positive. Map should be lower than 1. Actually it is 0.5, but the exact value depends
    on where we are sampling (i.e. recall's values)
    """
    gts = [
        dict(boxes=torch.Tensor([[10, 20, 15, 25]]), labels=torch.IntTensor([0])),
        dict(boxes=torch.Tensor([]), labels=torch.IntTensor([])),
    ]
    preds = [
        dict(boxes=torch.Tensor([[10, 20, 15, 25]]), scores=torch.Tensor([0.9]), labels=torch.IntTensor([0])),
        dict(boxes=torch.Tensor([[10, 20, 15, 25]]), scores=torch.Tensor([0.95]), labels=torch.IntTensor([0])),
    ]

    metric = MeanAveragePrecision()
    metric.update(preds, gts)
    result = metric.compute()
    assert result["map"] < 1, "MAP cannot be 1, as there is an image with no ground truth, but some predictions."


@pytest.mark.skipif(_pytest_condition, reason="test requires that torchvision=>0.8.0 is installed")
def test_error_on_wrong_input():
    """Test class input validation."""
    metric = MeanAveragePrecision()

    metric.update([], [])  # no error

    with pytest.raises(ValueError, match="Expected argument `preds` to be of type Sequence"):
        metric.update(torch.Tensor(), [])  # type: ignore

    with pytest.raises(ValueError, match="Expected argument `target` to be of type Sequence"):
        metric.update([], torch.Tensor())  # type: ignore

    with pytest.raises(ValueError, match="Expected argument `preds` and `target` to have the same length"):
        metric.update([dict()], [dict(), dict()])

    with pytest.raises(ValueError, match="Expected all dicts in `preds` to contain the `boxes` key"):
        metric.update(
            [dict(scores=torch.Tensor(), labels=torch.IntTensor)],
            [dict(boxes=torch.Tensor(), labels=torch.IntTensor())],
        )

    with pytest.raises(ValueError, match="Expected all dicts in `preds` to contain the `scores` key"):
        metric.update(
            [dict(boxes=torch.Tensor(), labels=torch.IntTensor)],
            [dict(boxes=torch.Tensor(), labels=torch.IntTensor())],
        )

    with pytest.raises(ValueError, match="Expected all dicts in `preds` to contain the `labels` key"):
        metric.update(
            [dict(boxes=torch.Tensor(), scores=torch.IntTensor)],
            [dict(boxes=torch.Tensor(), labels=torch.IntTensor())],
        )

    with pytest.raises(ValueError, match="Expected all dicts in `target` to contain the `boxes` key"):
        metric.update(
            [dict(boxes=torch.Tensor(), scores=torch.IntTensor, labels=torch.IntTensor)],
            [dict(labels=torch.IntTensor())],
        )

    with pytest.raises(ValueError, match="Expected all dicts in `target` to contain the `labels` key"):
        metric.update(
            [dict(boxes=torch.Tensor(), scores=torch.IntTensor, labels=torch.IntTensor)],
            [dict(boxes=torch.IntTensor())],
        )

    with pytest.raises(ValueError, match="Expected all boxes in `preds` to be of type Tensor"):
        metric.update(
            [dict(boxes=[], scores=torch.Tensor(), labels=torch.IntTensor())],
            [dict(boxes=torch.Tensor(), labels=torch.IntTensor())],
        )

    with pytest.raises(ValueError, match="Expected all scores in `preds` to be of type Tensor"):
        metric.update(
            [dict(boxes=torch.Tensor(), scores=[], labels=torch.IntTensor())],
            [dict(boxes=torch.Tensor(), labels=torch.IntTensor())],
        )

    with pytest.raises(ValueError, match="Expected all labels in `preds` to be of type Tensor"):
        metric.update(
            [dict(boxes=torch.Tensor(), scores=torch.Tensor(), labels=[])],
            [dict(boxes=torch.Tensor(), labels=torch.IntTensor())],
        )

    with pytest.raises(ValueError, match="Expected all boxes in `target` to be of type Tensor"):
        metric.update(
            [dict(boxes=torch.Tensor(), scores=torch.Tensor(), labels=torch.IntTensor())],
            [dict(boxes=[], labels=torch.IntTensor())],
        )

    with pytest.raises(ValueError, match="Expected all labels in `target` to be of type Tensor"):
        metric.update(
            [dict(boxes=torch.Tensor(), scores=torch.Tensor(), labels=torch.IntTensor())],
            [dict(boxes=torch.Tensor(), labels=[])],
        )
