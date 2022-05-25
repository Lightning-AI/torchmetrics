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

import json
from collections import namedtuple

import numpy as np
import pytest
import torch
from pycocotools import mask
from torch import IntTensor, Tensor

from tests.detection import _SAMPLE_DETECTION_SEGMENTATION
from tests.helpers.testers import MetricTester
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_8

Input = namedtuple("Input", ["preds", "target"])

with open(_SAMPLE_DETECTION_SEGMENTATION) as fp:
    inputs_json = json.load(fp)

_mask_unsqueeze_bool = lambda m: Tensor(mask.decode(m)).unsqueeze(0).bool()
_masks_stack_bool = lambda ms: Tensor(np.stack([mask.decode(m) for m in ms])).bool()

_inputs_masks = Input(
    preds=[
        [
            dict(masks=_mask_unsqueeze_bool(inputs_json["preds"][0]), scores=Tensor([0.236]), labels=IntTensor([4])),
            dict(
                masks=_masks_stack_bool([inputs_json["preds"][1], inputs_json["preds"][2]]),
                scores=Tensor([0.318, 0.726]),
                labels=IntTensor([3, 2]),
            ),  # 73
        ],
    ],
    target=[
        [
            dict(masks=_mask_unsqueeze_bool(inputs_json["targets"][0]), labels=IntTensor([4])),  # 42
            dict(
                masks=_masks_stack_bool([inputs_json["targets"][1], inputs_json["targets"][2]]),
                labels=IntTensor([2, 2]),
            ),  # 73
        ],
    ],
)


_inputs = Input(
    preds=[
        [
            dict(
                boxes=Tensor([[258.15, 41.29, 606.41, 285.07]]),
                scores=Tensor([0.236]),
                labels=IntTensor([4]),
            ),  # coco image id 42
            dict(
                boxes=Tensor([[61.00, 22.75, 565.00, 632.42], [12.66, 3.32, 281.26, 275.23]]),
                scores=Tensor([0.318, 0.726]),
                labels=IntTensor([3, 2]),
            ),  # coco image id 73
        ],
        [
            dict(
                boxes=Tensor(
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
                scores=Tensor([0.546, 0.3, 0.407, 0.611, 0.335, 0.805, 0.953]),
                labels=IntTensor([4, 1, 0, 0, 0, 0, 0]),
            ),  # coco image id 74
            dict(
                boxes=Tensor([[0.00, 2.87, 601.00, 421.52]]),
                scores=Tensor([0.699]),
                labels=IntTensor([5]),
            ),  # coco image id 133
        ],
    ],
    target=[
        [
            dict(
                boxes=Tensor([[214.1500, 41.2900, 562.4100, 285.0700]]),
                labels=IntTensor([4]),
            ),  # coco image id 42
            dict(
                boxes=Tensor(
                    [
                        [13.00, 22.75, 548.98, 632.42],
                        [1.66, 3.32, 270.26, 275.23],
                    ]
                ),
                labels=IntTensor([2, 2]),
            ),  # coco image id 73
        ],
        [
            dict(
                boxes=Tensor(
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
                labels=IntTensor([4, 1, 0, 0, 0, 0, 0]),
            ),  # coco image id 74
            dict(
                boxes=Tensor([[13.99, 2.87, 640.00, 421.52]]),
                labels=IntTensor([5]),
            ),  # coco image id 133
        ],
    ],
)

# example from this issue https://github.com/PyTorchLightning/metrics/issues/943
_inputs2 = Input(
    preds=[
        [
            dict(
                boxes=Tensor([[258.0, 41.0, 606.0, 285.0]]),
                scores=Tensor([0.536]),
                labels=IntTensor([0]),
            ),
        ],
        [
            dict(
                boxes=Tensor([[258.0, 41.0, 606.0, 285.0]]),
                scores=Tensor([0.536]),
                labels=IntTensor([0]),
            )
        ],
    ],
    target=[
        [
            dict(
                boxes=Tensor([[214.0, 41.0, 562.0, 285.0]]),
                labels=IntTensor([0]),
            )
        ],
        [
            dict(
                boxes=Tensor([]),
                labels=IntTensor([]),
            )
        ],
    ],
)

# Test empty preds case, to ensure bool inputs are properly casted to uint8
# From https://github.com/PyTorchLightning/metrics/issues/981
_inputs3 = Input(
    preds=[
        [
            dict(boxes=Tensor([]), scores=Tensor([]), labels=Tensor([])),
        ],
    ],
    target=[
        [
            dict(
                boxes=Tensor([[1.0, 2.0, 3.0, 4.0]]),
                scores=Tensor([0.8]),
                labels=Tensor([1]),
            ),
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
        "map": Tensor([0.706]),
        "map_50": Tensor([0.901]),
        "map_75": Tensor([0.846]),
        "map_small": Tensor([0.689]),
        "map_medium": Tensor([0.800]),
        "map_large": Tensor([0.701]),
        "mar_1": Tensor([0.592]),
        "mar_10": Tensor([0.716]),
        "mar_100": Tensor([0.716]),
        "mar_small": Tensor([0.767]),
        "mar_medium": Tensor([0.800]),
        "mar_large": Tensor([0.700]),
        "map_per_class": Tensor([0.725, 0.800, 0.454, -1.000, 0.650, 0.900]),
        "mar_100_per_class": Tensor([0.780, 0.800, 0.450, -1.000, 0.650, 0.900]),
    }


def _compare_fn_segm(preds, target) -> dict:
    """Comparison function for map implementation for instance segmentation.

    Official pycocotools results calculated from a subset of https://github.com/cocodataset/cocoapi/tree/master/results
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.352
        Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.752
        Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.252
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.352
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.350
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.350
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.350
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.350
    """
    return {
        "map": Tensor([0.352]),
        "map_50": Tensor([0.742]),
        "map_75": Tensor([0.252]),
        "map_small": Tensor([-1]),
        "map_medium": Tensor([-1]),
        "map_large": Tensor([0.352]),
        "mar_1": Tensor([0.35]),
        "mar_10": Tensor([0.35]),
        "mar_100": Tensor([0.35]),
        "mar_small": Tensor([-1]),
        "mar_medium": Tensor([-1]),
        "mar_large": Tensor([0.35]),
        "map_per_class": Tensor([0.4039604, -1.0, 0.3]),
        "mar_100_per_class": Tensor([0.4, -1.0, 0.3]),
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
    def test_map_bbox(self, compute_on_cpu, ddp):

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

    @pytest.mark.parametrize("ddp", [False])
    def test_map_segm(self, compute_on_cpu, ddp):
        """Test modular implementation for correctness."""

        self.run_class_metric_test(
            ddp=ddp,
            preds=_inputs_masks.preds,
            target=_inputs_masks.target,
            metric_class=MeanAveragePrecision,
            sk_metric=_compare_fn_segm,
            dist_sync_on_step=False,
            check_batch=False,
            metric_args={"class_metrics": True, "compute_on_cpu": compute_on_cpu, "iou_type": "segm"},
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
            dict(boxes=Tensor([]), scores=Tensor([]), labels=IntTensor([])),
        ],
        [
            dict(boxes=Tensor([[214.1500, 41.2900, 562.4100, 285.0700]]), labels=IntTensor([4])),
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
                boxes=Tensor([[214.1500, 41.2900, 562.4100, 285.0700]]),
                scores=Tensor([0.5]),
                labels=IntTensor([4]),
            ),
        ],
        [
            dict(boxes=Tensor([]), labels=IntTensor([])),
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
@pytest.mark.parametrize("inputs", [_inputs, _inputs2, _inputs3])
def test_map_gpu(inputs):
    """Test predictions on single gpu."""
    metric = MeanAveragePrecision()
    metric = metric.to("cuda")
    for preds, targets in zip(inputs.preds, inputs.target):
        metric.update(_move_to_gpu(preds), _move_to_gpu(targets))
    metric.compute()


@pytest.mark.skipif(_pytest_condition, reason="test requires that torchvision=>0.8.0 is installed")
@pytest.mark.skipif(_gpu_test_condition, reason="test requires CUDA availability")
def test_map_with_custom_thresholds():
    """Test that map works with custom iou thresholds."""
    metric = MeanAveragePrecision(iou_thresholds=[0.1, 0.2])
    metric = metric.to("cuda")
    for preds, targets in zip(_inputs.preds, _inputs.target):
        metric.update(_move_to_gpu(preds), _move_to_gpu(targets))
    res = metric.compute()
    assert res["map_50"].item() == -1
    assert res["map_75"].item() == -1


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
        dict(boxes=Tensor([[10, 20, 15, 25]]), labels=IntTensor([0])),
        dict(boxes=Tensor([[10, 20, 15, 25]]), labels=IntTensor([0])),
    ]
    preds = [
        dict(boxes=Tensor([[10, 20, 15, 25]]), scores=Tensor([0.9]), labels=IntTensor([0])),
        # Empty prediction
        dict(boxes=Tensor([]), scores=Tensor([]), labels=IntTensor([])),
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
        dict(boxes=Tensor([[10, 20, 15, 25]]), labels=IntTensor([0])),
        dict(boxes=Tensor([]), labels=IntTensor([])),
    ]
    preds = [
        dict(boxes=Tensor([[10, 20, 15, 25]]), scores=Tensor([0.9]), labels=IntTensor([0])),
        dict(boxes=Tensor([[10, 20, 15, 25]]), scores=Tensor([0.95]), labels=IntTensor([0])),
    ]

    metric = MeanAveragePrecision()
    metric.update(preds, gts)
    result = metric.compute()
    assert result["map"] < 1, "MAP cannot be 1, as there is an image with no ground truth, but some predictions."


@pytest.mark.skipif(_pytest_condition, reason="test requires that torchvision=>0.8.0 is installed")
def test_segm_iou_empty_mask():
    """Test empty ground truths."""
    metric = MeanAveragePrecision(iou_type="segm")

    metric.update(
        [
            dict(
                masks=torch.randint(0, 1, (1, 10, 10)).bool(),
                scores=Tensor([0.5]),
                labels=IntTensor([4]),
            ),
        ],
        [
            dict(masks=Tensor([]), labels=IntTensor([])),
        ],
    )

    metric.compute()


@pytest.mark.skipif(_pytest_condition, reason="test requires that torchvision=>0.8.0 is installed")
def test_error_on_wrong_input():
    """Test class input validation."""
    metric = MeanAveragePrecision()

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
