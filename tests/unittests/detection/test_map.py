# Copyright The Lightning team.
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
from copy import deepcopy
from functools import partial

import numpy as np
import pytest
import torch
from pycocotools import mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch import IntTensor, Tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.utilities.imports import _PYCOCOTOOLS_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_8

from unittests.detection import _DETECTION_BBOX, _DETECTION_SEGM, _DETECTION_VAL, _SAMPLE_DETECTION_SEGMENTATION
from unittests.helpers.testers import MetricTester


def _generate_inputs(iou_type):
    """Generates inputs for the MAP metric."""
    gt = COCO(_DETECTION_VAL)
    dt = gt.loadRes(_DETECTION_BBOX if iou_type == "bbox" else _DETECTION_SEGM)
    img_ids = sorted(gt.getImgIds())
    img_ids = img_ids[0:100]

    gt_dataset = gt.dataset["annotations"]
    dt_dataset = dt.dataset["annotations"]

    preds = {}
    for p in dt_dataset:
        if p["image_id"] not in preds:
            preds[p["image_id"]] = {"boxes" if iou_type == "bbox" else "masks": [], "scores": [], "labels": []}
        if iou_type == "bbox":
            preds[p["image_id"]]["boxes"].append(p["bbox"])
        else:
            preds[p["image_id"]]["masks"].append(gt.annToMask(p))
        preds[p["image_id"]]["scores"].append(p["score"])
        preds[p["image_id"]]["labels"].append(p["category_id"])
    missing_pred = set(img_ids) - set(preds.keys())
    for i in missing_pred:
        preds[i] = {"boxes" if iou_type == "bbox" else "masks": [], "scores": [], "labels": []}

    target = {}
    for t in gt_dataset:
        if t["image_id"] not in img_ids:
            continue
        if t["image_id"] not in target:
            target[t["image_id"]] = {"boxes" if iou_type == "bbox" else "masks": [], "labels": []}
        if iou_type == "bbox":
            target[t["image_id"]]["boxes"].append(t["bbox"])
        else:
            target[t["image_id"]]["masks"].append(gt.annToMask(t))
        target[t["image_id"]]["labels"].append(t["category_id"])

    if iou_type == "bbox":
        preds = [
            {
                "boxes": torch.tensor(p["boxes"]),
                "scores": torch.tensor(p["scores"]),
                "labels": torch.tensor(p["labels"]),
            }
            for p in preds.values()
        ]
        target = [{"boxes": torch.tensor(t["boxes"]), "labels": torch.tensor(t["labels"])} for t in target.values()]
    else:
        preds = [
            {
                "masks": torch.tensor(p["masks"]),
                "scores": torch.tensor(p["scores"]),
                "labels": torch.tensor(p["labels"]),
            }
            for p in preds.values()
        ]
        target = [{"masks": torch.tensor(t["masks"]), "labels": torch.tensor(t["labels"])} for t in target.values()]

    # create 10 batches of 10 preds/targets each
    preds = [preds[10 * i : 10 * (i + 1)] for i in range(10)]
    target = [target[10 * i : 10 * (i + 1)] for i in range(10)]

    return preds, target


_bbox_input = _generate_inputs("bbox")
_segm_input = _generate_inputs("segm")


def _compare_fn(preds, target, iou_type, class_metrics=True):
    """Taken from https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb."""
    gt = COCO(_DETECTION_VAL)
    dt = gt.loadRes(_DETECTION_BBOX if iou_type == "bbox" else _DETECTION_SEGM)
    img_ids = sorted(gt.getImgIds())
    img_ids = img_ids[0:100]
    coco_eval = COCOeval(gt, dt, iou_type)
    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    global_stats = deepcopy(coco_eval.stats)

    map_per_class_values = torch.Tensor([-1])
    mar_100_per_class_values = torch.Tensor([-1])
    classes = Tensor(np.unique([x["category_id"] for x in gt.dataset["annotations"]]))
    if class_metrics:
        map_per_class_list = []
        mar_100_per_class_list = []
        for class_id in classes.tolist():
            coco_eval.params.catIds = [class_id]
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            class_stats = coco_eval.stats
            map_per_class_list.append(torch.Tensor([class_stats[0]]))
            mar_100_per_class_list.append(torch.Tensor([class_stats[8]]))

        map_per_class_values = torch.Tensor(map_per_class_list)
        mar_100_per_class_values = torch.Tensor(mar_100_per_class_list)

    return {
        "map": Tensor([global_stats[0]]),
        "map_50": Tensor([global_stats[1]]),
        "map_75": Tensor([global_stats[2]]),
        "map_small": Tensor([global_stats[3]]),
        "map_medium": Tensor([global_stats[4]]),
        "map_large": Tensor([global_stats[5]]),
        "mar_1": Tensor([global_stats[6]]),
        "mar_10": Tensor([global_stats[7]]),
        "mar_100": Tensor([global_stats[8]]),
        "mar_small": Tensor([global_stats[9]]),
        "mar_medium": Tensor([global_stats[10]]),
        "mar_large": Tensor([global_stats[11]]),
        "map_per_class": map_per_class_values,
        "mar_100_per_class": mar_100_per_class_values,
        "classes": classes,
    }


_pytest_condition = not (_PYCOCOTOOLS_AVAILABLE and _TORCHVISION_GREATER_EQUAL_0_8)


@pytest.mark.skipif(_pytest_condition, reason="test requires that torchvision=>0.8.0 and pycocotools is installed")
@pytest.mark.parametrize("iou_type", ["bbox", "segm"])
class TestMAPNew(MetricTester):
    """Test map metric."""

    # @pytest.mark.parametrize("ddp", [False, True])
    def test_map(self, iou_type):
        """Test modular implementation for correctness."""
        preds, target = _segm_input if iou_type == "segm" else _bbox_input
        self.run_class_metric_test(
            ddp=False,
            preds=preds,
            target=target,
            metric_class=MeanAveragePrecision,
            reference_metric=partial(_compare_fn, iou_type=iou_type),
            metric_args={"iou_type": iou_type},
            check_batch=False,
        )


Input = namedtuple("Input", ["preds", "target"])


def _create_inputs_masks() -> Input:
    with open(_SAMPLE_DETECTION_SEGMENTATION) as fp:
        inputs_json = json.load(fp)

    _mask_unsqueeze_bool = lambda m: Tensor(mask.decode(m)).unsqueeze(0).bool()
    _masks_stack_bool = lambda ms: Tensor(np.stack([mask.decode(m) for m in ms])).bool()

    return Input(
        preds=[
            [
                {
                    "masks": _mask_unsqueeze_bool(inputs_json["preds"][0]),
                    "scores": Tensor([0.236]),
                    "labels": IntTensor([4]),
                },
                {
                    "masks": _masks_stack_bool([inputs_json["preds"][1], inputs_json["preds"][2]]),
                    "scores": Tensor([0.318, 0.726]),
                    "labels": IntTensor([3, 2]),
                },  # 73
            ],
            [
                {
                    "masks": _mask_unsqueeze_bool(inputs_json["preds"][0]),
                    "scores": Tensor([0.236]),
                    "labels": IntTensor([4]),
                },
                {
                    "masks": _masks_stack_bool([inputs_json["preds"][1], inputs_json["preds"][2]]),
                    "scores": Tensor([0.318, 0.726]),
                    "labels": IntTensor([3, 2]),
                },  # 73
            ],
        ],
        target=[
            [
                {"masks": _mask_unsqueeze_bool(inputs_json["targets"][0]), "labels": IntTensor([4])},  # 42
                {
                    "masks": _masks_stack_bool([inputs_json["targets"][1], inputs_json["targets"][2]]),
                    "labels": IntTensor([2, 2]),
                },  # 73
            ],
            [
                {"masks": _mask_unsqueeze_bool(inputs_json["targets"][0]), "labels": IntTensor([4])},  # 42
                {
                    "masks": _masks_stack_bool([inputs_json["targets"][1], inputs_json["targets"][2]]),
                    "labels": IntTensor([2, 2]),
                },  # 73
            ],
        ],
    )


_inputs = Input(
    preds=[
        [
            {
                "boxes": Tensor([[258.15, 41.29, 606.41, 285.07]]),
                "scores": Tensor([0.236]),
                "labels": IntTensor([4]),
            },  # coco image id 42
            {
                "boxes": Tensor([[61.00, 22.75, 565.00, 632.42], [12.66, 3.32, 281.26, 275.23]]),
                "scores": Tensor([0.318, 0.726]),
                "labels": IntTensor([3, 2]),
            },  # coco image id 73
        ],
        [
            {
                "boxes": Tensor(
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
                "scores": Tensor([0.546, 0.3, 0.407, 0.611, 0.335, 0.805, 0.953]),
                "labels": IntTensor([4, 1, 0, 0, 0, 0, 0]),
            },  # coco image id 74
            {
                "boxes": Tensor(
                    [
                        [72.92, 45.96, 91.23, 80.57],
                        [45.17, 45.34, 66.28, 79.83],
                        [82.28, 47.04, 99.66, 78.50],
                        [59.96, 46.17, 80.35, 80.48],
                        [75.29, 23.01, 91.85, 50.85],
                        [71.14, 1.10, 96.96, 28.33],
                        [61.34, 55.23, 77.14, 79.57],
                        [41.17, 45.78, 60.99, 78.48],
                        [56.18, 44.80, 64.42, 56.25],
                    ]
                ),
                "scores": Tensor([0.532, 0.204, 0.782, 0.202, 0.883, 0.271, 0.561, 0.204, 0.349]),
                "labels": IntTensor([49, 49, 49, 49, 49, 49, 49, 49, 49]),
            },  # coco image id 987 category_id 49
        ],
    ],
    target=[
        [
            {
                "boxes": Tensor([[214.1500, 41.2900, 562.4100, 285.0700]]),
                "labels": IntTensor([4]),
            },  # coco image id 42
            {
                "boxes": Tensor(
                    [
                        [13.00, 22.75, 548.98, 632.42],
                        [1.66, 3.32, 270.26, 275.23],
                    ]
                ),
                "labels": IntTensor([2, 2]),
            },  # coco image id 73
        ],
        [
            {
                "boxes": Tensor(
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
                "labels": IntTensor([4, 1, 0, 0, 0, 0, 0]),
            },  # coco image id 74
            {
                "boxes": Tensor(
                    [
                        [72.92, 45.96, 91.23, 80.57],
                        [50.17, 45.34, 71.28, 79.83],
                        [81.28, 47.04, 98.66, 78.50],
                        [63.96, 46.17, 84.35, 80.48],
                        [75.29, 23.01, 91.85, 50.85],
                        [56.39, 21.65, 75.66, 45.54],
                        [73.14, 1.10, 98.96, 28.33],
                        [62.34, 55.23, 78.14, 79.57],
                        [44.17, 45.78, 63.99, 78.48],
                        [58.18, 44.80, 66.42, 56.25],
                    ]
                ),
                "labels": IntTensor([49, 49, 49, 49, 49, 49, 49, 49, 49, 49]),
            },  # coco image id 987 category_id 49
        ],
    ],
)

# example from this issue https://github.com/Lightning-AI/torchmetrics/issues/943
_inputs2 = Input(
    preds=[
        [
            {
                "boxes": Tensor([[258.0, 41.0, 606.0, 285.0]]),
                "scores": Tensor([0.536]),
                "labels": IntTensor([0]),
            },
        ],
        [
            {
                "boxes": Tensor([[258.0, 41.0, 606.0, 285.0]]),
                "scores": Tensor([0.536]),
                "labels": IntTensor([0]),
            }
        ],
    ],
    target=[
        [
            {
                "boxes": Tensor([[214.0, 41.0, 562.0, 285.0]]),
                "labels": IntTensor([0]),
            }
        ],
        [
            {
                "boxes": Tensor([]),
                "labels": IntTensor([]),
            }
        ],
    ],
)

# Test empty preds case, to ensure bool inputs are properly casted to uint8
# From https://github.com/Lightning-AI/torchmetrics/issues/981
# and https://github.com/Lightning-AI/torchmetrics/issues/1147
_inputs3 = Input(
    preds=[
        [
            {
                "boxes": Tensor([[258.0, 41.0, 606.0, 285.0]]),
                "scores": Tensor([0.536]),
                "labels": IntTensor([0]),
            },
        ],
        [
            {"boxes": Tensor([]), "scores": Tensor([]), "labels": Tensor([])},
        ],
    ],
    target=[
        [
            {
                "boxes": Tensor([[214.0, 41.0, 562.0, 285.0]]),
                "labels": IntTensor([0]),
            }
        ],
        [
            {
                "boxes": Tensor([[1.0, 2.0, 3.0, 4.0]]),
                "scores": Tensor([0.8]),  # target does not have scores
                "labels": IntTensor([1]),
            },
        ],
    ],
)


_inputs4 = Input(
    preds=[
        [
            {
                "boxes": torch.Tensor([[258.15, 41.29, 606.41, 285.07]]),
                "scores": torch.Tensor([0.236]),
                "labels": torch.IntTensor([4]),
            },  # coco image id 42
            {
                "boxes": torch.Tensor([[61.00, 22.75, 565.00, 632.42], [12.66, 3.32, 281.26, 275.23]]),
                "scores": torch.Tensor([0.318, 0.726]),
                "labels": torch.IntTensor([3, 2]),
            },  # coco image id 73
        ],
        [
            {
                "boxes": torch.Tensor(
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
                "scores": torch.Tensor([0.546, 0.3, 0.407, 0.611, 0.335, 0.805, 0.953]),
                "labels": torch.IntTensor([4, 1, 0, 0, 0, 0, 0]),
            },  # coco image id 74
            {
                "boxes": torch.Tensor([[0.00, 2.87, 601.00, 421.52]]),
                "scores": torch.Tensor([0.423]),
                "labels": torch.IntTensor([5]),
            },  # coco image id 133
        ],
    ],
    target=[
        [
            {
                "boxes": torch.Tensor([[214.1500, 41.2900, 562.4100, 285.0700]]),
                "labels": torch.IntTensor([4]),
            },  # coco image id 42
            {
                "boxes": torch.Tensor(
                    [
                        [13.00, 22.75, 548.98, 632.42],
                        [1.66, 3.32, 270.26, 275.23],
                    ]
                ),
                "labels": torch.IntTensor([2, 2]),
            },  # coco image id 73
        ],
        [
            {
                "boxes": torch.Tensor(
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
                "labels": torch.IntTensor([4, 1, 0, 0, 0, 0, 0]),
            },  # coco image id 74
            {
                "boxes": torch.Tensor([[13.99, 2.87, 640.00, 421.52]]),
                "labels": torch.IntTensor([5]),
            },  # coco image id 133
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
        "map_50": Tensor([0.752]),
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
        "classes": Tensor([2, 3, 4]),
    }


_pytest_condition = not _TORCHVISION_GREATER_EQUAL_0_8


@pytest.mark.skipif(_pytest_condition, reason="test requires that torchvision=>0.8.0 is installed")
class TestMAP(MetricTester):
    """Test the MAP metric for object detection predictions.

    Results are compared to original values from the pycocotools implementation. A subset of the first 10 fake
    predictions of the official repo is used:
    https://github.com/cocodataset/cocoapi/blob/master/results/instances_val2014_fakebbox100_results.json
    """

    atol = 1e-2

    @pytest.mark.parametrize("ddp", [False, True])
    def test_map_bbox(self, ddp):
        """Test modular implementation for correctness."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=_inputs4.preds,
            target=_inputs4.target,
            metric_class=MeanAveragePrecision,
            reference_metric=_compare_fn,
            check_batch=False,
            metric_args={"class_metrics": True},
        )

    @pytest.mark.parametrize("ddp", [False, True])
    def test_map_segm(self, ddp):
        """Test modular implementation for correctness."""
        _inputs_masks = _create_inputs_masks()
        self.run_class_metric_test(
            ddp=ddp,
            preds=_inputs_masks.preds,
            target=_inputs_masks.target,
            metric_class=MeanAveragePrecision,
            reference_metric=_compare_fn_segm,
            check_batch=False,
            metric_args={"class_metrics": True, "iou_type": "segm"},
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
        [{"boxes": Tensor([]), "scores": Tensor([]), "labels": IntTensor([])}],
        [{"boxes": Tensor([[214.1500, 41.2900, 562.4100, 285.0700]]), "labels": IntTensor([4])}],
    )
    metric.compute()


@pytest.mark.skipif(_pytest_condition, reason="test requires that torchvision=>0.8.0 is installed")
def test_empty_ground_truths():
    """Test empty ground truths."""
    metric = MeanAveragePrecision()

    metric.update(
        [
            {
                "boxes": Tensor([[214.1500, 41.2900, 562.4100, 285.0700]]),
                "scores": Tensor([0.5]),
                "labels": IntTensor([4]),
            }
        ],
        [{"boxes": Tensor([]), "labels": IntTensor([])}],
    )
    metric.compute()


@pytest.mark.skipif(_pytest_condition, reason="test requires that torchvision=>0.8.0 is installed")
def test_empty_ground_truths_xywh():
    """Test empty ground truths in xywh format."""
    metric = MeanAveragePrecision(box_format="xywh")

    metric.update(
        [
            {
                "boxes": Tensor([[214.1500, 41.2900, 348.2600, 243.7800]]),
                "scores": Tensor([0.5]),
                "labels": IntTensor([4]),
            }
        ],
        [{"boxes": Tensor([]), "labels": IntTensor([])}],
    )
    metric.compute()


@pytest.mark.skipif(_pytest_condition, reason="test requires that torchvision=>0.8.0 is installed")
def test_empty_preds_xywh():
    """Test empty predictions in xywh format."""
    metric = MeanAveragePrecision(box_format="xywh")

    metric.update(
        [{"boxes": Tensor([]), "scores": Tensor([]), "labels": IntTensor([])}],
        [{"boxes": Tensor([[214.1500, 41.2900, 348.2600, 243.7800]]), "labels": IntTensor([4])}],
    )
    metric.compute()


@pytest.mark.skipif(_pytest_condition, reason="test requires that torchvision=>0.8.0 is installed")
def test_empty_ground_truths_cxcywh():
    """Test empty ground truths in cxcywh format."""
    metric = MeanAveragePrecision(box_format="cxcywh")

    metric.update(
        [
            {
                "boxes": Tensor([[388.2800, 163.1800, 348.2600, 243.7800]]),
                "scores": Tensor([0.5]),
                "labels": IntTensor([4]),
            }
        ],
        [{"boxes": Tensor([]), "labels": IntTensor([])}],
    )
    metric.compute()


@pytest.mark.skipif(_pytest_condition, reason="test requires that torchvision=>0.8.0 is installed")
def test_empty_preds_cxcywh():
    """Test empty predictions in cxcywh format."""
    metric = MeanAveragePrecision(box_format="cxcywh")

    metric.update(
        [{"boxes": Tensor([]), "scores": Tensor([]), "labels": IntTensor([])}],
        [{"boxes": Tensor([[388.2800, 163.1800, 348.2600, 243.7800]]), "labels": IntTensor([4])}],
    )
    metric.compute()


_gpu_test_condition = not torch.cuda.is_available()


def _move_to_gpu(inputs):
    for x in inputs:
        for key in x:
            if torch.is_tensor(x[key]):
                x[key] = x[key].to("cuda")
    return inputs


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
        {"boxes": Tensor([[10, 20, 15, 25]]), "labels": IntTensor([0])},
        {"boxes": Tensor([[10, 20, 15, 25]]), "labels": IntTensor([0])},
    ]
    preds = [
        {"boxes": Tensor([[10, 20, 15, 25]]), "scores": Tensor([0.9]), "labels": IntTensor([0])},
        # Empty prediction
        {"boxes": Tensor([]), "scores": Tensor([]), "labels": IntTensor([])},
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
        {"boxes": Tensor([[10, 20, 15, 25]]), "labels": IntTensor([0])},
        {"boxes": Tensor([]), "labels": IntTensor([])},
    ]
    preds = [
        {"boxes": Tensor([[10, 20, 15, 25]]), "scores": Tensor([0.9]), "labels": IntTensor([0])},
        {"boxes": Tensor([[10, 20, 15, 25]]), "scores": Tensor([0.95]), "labels": IntTensor([0])},
    ]

    metric = MeanAveragePrecision()
    metric.update(preds, gts)
    result = metric.compute()
    assert result["map"] < 1, "MAP cannot be 1, as there is an image with no ground truth, but some predictions."


@pytest.mark.skipif(_pytest_condition, reason="test requires that torchvision=>0.8.0 is installed")
def test_segm_iou_empty_gt_mask():
    """Test empty ground truths."""
    metric = MeanAveragePrecision(iou_type="segm")

    metric.update(
        [{"masks": torch.randint(0, 1, (1, 10, 10)).bool(), "scores": Tensor([0.5]), "labels": IntTensor([4])}],
        [{"masks": Tensor([]), "labels": IntTensor([])}],
    )

    metric.compute()


@pytest.mark.skipif(_pytest_condition, reason="test requires that torchvision=>0.8.0 is installed")
def test_segm_iou_empty_pred_mask():
    """Test empty predictions."""
    metric = MeanAveragePrecision(iou_type="segm")

    metric.update(
        [{"masks": torch.BoolTensor([]), "scores": Tensor([]), "labels": IntTensor([])}],
        [{"masks": torch.randint(0, 1, (1, 10, 10)).bool(), "labels": IntTensor([4])}],
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


def _generate_random_segm_input(device):
    """Generate random inputs for mAP when iou_type=segm."""
    preds = []
    targets = []
    for _ in range(2):
        result = {}
        num_preds = torch.randint(0, 10, (1,)).item()
        result["scores"] = torch.rand((num_preds,), device=device)
        result["labels"] = torch.randint(0, 10, (num_preds,), device=device)
        result["masks"] = torch.randint(0, 2, (num_preds, 10, 10), device=device).bool()
        preds.append(result)
        gt = {}
        num_gt = torch.randint(0, 10, (1,)).item()
        gt["labels"] = torch.randint(0, 10, (num_gt,), device=device)
        gt["masks"] = torch.randint(0, 2, (num_gt, 10, 10), device=device).bool()
        targets.append(gt)
    return preds, targets


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
def test_device_changing():
    """See issue: https://github.com/Lightning-AI/torchmetrics/issues/1743.

    Checks that the custom apply function of the metric works as expected.
    """
    device = "cuda"
    metric = MeanAveragePrecision(iou_type="segm").to(device)

    for _ in range(2):
        preds, targets = _generate_random_segm_input(device)
        metric.update(preds, targets)

    metric = metric.cpu()
    val = metric.compute()
    assert isinstance(val, dict)


def test_order():
    """Test that the ordering of input does not matter.

    Issue: https://github.com/Lightning-AI/torchmetrics/issues/1774
    """
    targets = [
        {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.long),
        },
        {
            "boxes": torch.FloatTensor([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]]),
            "labels": torch.LongTensor([1, 2]),
        },
    ]

    preds = [
        {
            "boxes": torch.FloatTensor([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]]),
            "labels": torch.LongTensor([1, 2]),
            "scores": torch.FloatTensor([0.9, 0.8]),
        },
        {
            "boxes": torch.FloatTensor([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]]),
            "labels": torch.LongTensor([1, 2]),
            "scores": torch.FloatTensor([0.9, 0.8]),
        },
    ]
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    metrics = metric(preds, targets)
    assert metrics["map_50"] == torch.tensor([0.5])

    targets = [
        {
            "boxes": torch.FloatTensor([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]]),
            "labels": torch.LongTensor([1, 2]),
        },
        {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.long),
        },
    ]

    preds = [
        {
            "boxes": torch.FloatTensor([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]]),
            "labels": torch.LongTensor([1, 2]),
            "scores": torch.FloatTensor([0.9, 0.8]),
        },
        {
            "boxes": torch.FloatTensor([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]]),
            "labels": torch.LongTensor([1, 2]),
            "scores": torch.FloatTensor([0.9, 0.8]),
        },
    ]
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    metrics = metric(preds, targets)
    assert metrics["map_50"] == torch.tensor([0.5])


def test_corner_case():
    """Issue: https://github.com/Lightning-AI/torchmetrics/issues/1184."""
    metric = MeanAveragePrecision(iou_thresholds=[0.501], class_metrics=True)
    preds = [
        {
            "boxes": torch.Tensor(
                [[0, 0, 20, 20], [30, 30, 50, 50], [70, 70, 90, 90], [100, 100, 120, 120]]
            ),  # FP  # FP
            "scores": torch.Tensor([0.6, 0.6, 0.6, 0.6]),
            "labels": torch.IntTensor([0, 1, 2, 3]),
        }
    ]

    targets = [
        {
            "boxes": torch.Tensor([[0, 0, 20, 20], [30, 30, 50, 50]]),
            "labels": torch.IntTensor([0, 1]),
        }
    ]
    metric.update(preds, targets)
    res = metric.compute()
    assert res["map"] == torch.tensor([0.5])
