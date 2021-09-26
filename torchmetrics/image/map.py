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
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _PYCOCOTOOLS_AVAILABLE

if _PYCOCOTOOLS_AVAILABLE:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
else:
    COCO, COCOeval = None, None

COCO_STATS_MAP_VALUE_INDEX = 0
COCO_STATS_MAR_VALUE_INDEX = 8


@dataclass
class MAPMetricResults:
    """Dataclass to wrap the final mAP results."""

    map_value: Tensor
    mar_value: Tensor
    map_per_class_value: List[Tensor]
    mar_per_class_value: List[Tensor]


class _hide_prints:
    """Internal helper context to suppress the default output of the pycocotools package."""

    def __init__(self) -> None:
        self._original_stdout = None

    def __enter__(self) -> None:
        self._original_stdout = sys.stdout  # type: ignore
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore
        sys.stdout.close()
        sys.stdout = self._original_stdout  # type: ignore


def _input_validator(preds: List[Dict[str, torch.Tensor]], targets: List[Dict[str, torch.Tensor]]) -> None:
    """Ensure the correct input format of `preds` and `targets`"""

    if not isinstance(preds, list):
        raise ValueError("Expected argument `preds` to be of type List")
    if not isinstance(targets, list):
        raise ValueError("Expected argument `target` to be of type List")
    if len(preds) != len(targets):
        raise ValueError("Expected argument `preds` and `target` to have the same length")

    for k in ["boxes", "scores", "labels"]:
        if any(k not in p for p in preds):
            raise ValueError(f"Expected all dicts in `preds` to contain the `{k}` key")

    for k in ["boxes", "labels"]:
        if any(k not in p for p in targets):
            raise ValueError(f"Expected all dicts in `target` to contain the `{k}` key")

    if any(type(pred["boxes"]) is not torch.Tensor for pred in preds):
        raise ValueError("Expected all boxes in `preds` to be of type torch.Tensor")
    if any(type(pred["scores"]) is not torch.Tensor for pred in preds):
        raise ValueError("Expected all scores in `preds` to be of type torch.Tensor")
    if any(type(pred["labels"]) is not torch.Tensor for pred in preds):
        raise ValueError("Expected all labels in `preds` to be of type torch.Tensor")
    if any(type(target["boxes"]) is not torch.Tensor for target in targets):
        raise ValueError("Expected all boxes in `target` to be of type torch.Tensor")
    if any(type(target["labels"]) is not torch.Tensor for target in targets):
        raise ValueError("Expected all labels in `target` to be of type torch.Tensor")

    for i, k in enumerate(targets):
        if k["boxes"].size(0) != k["labels"].size(0):
            raise ValueError(
                f"Input boxes and labels of sample {i} in targets have a"
                f" different length (expected {k['boxes'].size(0)} labels, got {k['labels'].size(0)})"
            )
    for i, k in enumerate(preds):
        if k["boxes"].size(0) != k["labels"].size(0) != k["scores"].size(0):
            raise ValueError(
                f"Input boxes, labels and scores of sample {i} in preds have a"
                f" different length (expected {k['boxes'].size(0)} labels and scores,"
                f" got {k['labels'].size(0)} labels and {k['scores'].size(0)})"
            )


class MAP(Metric):
    r"""
    Computes the `Mean-Average-Precision (mAP) and Mean-Average-Recall (mAR)`_ for object detection predictions.
    Optionally, the mAP and mAR values can be calculated per class.

    Predicted boxes and targets have to be in Pascal VOC format
    (xmin-top left, ymin-top left, xmax-bottom right, ymax-bottom right).
    See the `update` function for more information about the input format to this metric.

    .. note::
        This metric is a wrapper for the
        `pycocotools <https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools>`_,
        which is a standard implementation for the mAP metric for object detection. Using this metric
        therefore requires you to have `pycocotools` installed. Please install with `pip install pycocotools` or
        `pip install torchmetrics[image]`

    .. note::
        As the pycocotools library cannot deal with tensors directly, all results have to be transfered
        to the CPU, this might have an performance impact on your training

    Args:
        num_classes:
            Number of classes, required for mAP values per class. default: 0 (deactivate)
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: False
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)

    Raises:
        ValueError:
            If ``pycocotools`` is not installed
        ValueError:
            If ``num_classes`` is not an integer larger or equal to 0
    """

    def __init__(
        self,
        num_classes: int = 0,
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )
        if not _PYCOCOTOOLS_AVAILABLE:
            raise ValueError(
                "`MAP` metric requires that `pycocotools` installed. Please install"
                " with `pip install pycocotools` or `pip install torchmetrics[image]`"
            )

        self.add_state("average_precision", default=torch.tensor(data=[], dtype=torch.float), dist_reduce_fx="mean")
        self.add_state("average_recall", default=torch.tensor(data=[], dtype=torch.float), dist_reduce_fx="mean")
        if not (isinstance(num_classes, int) and num_classes >= 0):
            raise ValueError("Expected argument `num_classes` to be a integer larger or equal to 0")
        self.num_classes = num_classes

        self.add_state("detection_boxes", default=[])
        self.add_state("detection_scores", default=[])
        self.add_state("detection_labels", default=[])
        self.add_state("groundtruth_boxes", default=[])
        self.add_state("groundtruth_labels", default=[])

        for class_id in range(num_classes):
            self.add_state(f"ap_{class_id}", default=torch.tensor(data=[], dtype=torch.float), dist_reduce_fx="mean")
            self.add_state(f"ar_{class_id}", default=torch.tensor(data=[], dtype=torch.float), dist_reduce_fx="mean")

    def update(self, preds: Any, target: Any) -> None:  # type: ignore
        """Updates mAP and mAR values with metric values from given predictions and groundtruth.

        Args:
            preds:
                A list consisting of dictionaries each containing the key-values
                (each dictionary corresponds to a single image):

                - ``boxes``: torch.FloatTensor of shape
                    [num_boxes, 4] containing `num_boxes` detection boxes of the format
                    [xmin, ymin, xmax, ymax] in absolute image coordinates.
                - ``scores``: torch.FloatTensor of shape
                    [num_boxes] containing detection scores for the boxes.
                - ``labels``: torch.IntTensor of shape
                    [num_boxes] containing 0-indexed detection classes for the boxes.

            target:
                A list consisting of dictionaries each containing the key-values
                (each dictionary corresponds to a single image):

                - ``boxes``: torch.FloatTensor of shape
                    [num_boxes, 4] containing `num_boxes` groundtruth boxes of the format
                    [xmin, ymin, xmax, ymax] in absolute image coordinates.
                - ``labels``: torch.IntTensor of shape
                    [num_boxes] containing 1-indexed groundtruth classes for the boxes.

        Raises:
            ValueError:
                If ``preds`` is not of type List[Dict[str, torch.Tensor]]
            ValueError:
                If ``target`` is not of type List[Dict[str, torch.Tensor]]
            ValueError:
                If `preds` and `target` are not of the same length
            ValueError:
                If any of `preds.boxes`, `preds.scores`
                and `preds.labels` are not of the same length
            ValueError:
                If any of `target.boxes` and `target.labels` are not of the same length
            ValueError:
                If any box is not type float and of length 4
            ValueError:
                If any class is not type int and of length 1
            ValueError:
                If any score is not type float and of length 1
        """
        _input_validator(preds, target)

        for item in preds:
            self.detection_boxes.append(item["boxes"])
            self.detection_scores.append(item["scores"])
            self.detection_labels.append(item["labels"])

        for item in target:
            self.groundtruth_boxes.append(item["boxes"])
            self.groundtruth_labels.append(item["labels"])

    def compute(self) -> MAPMetricResults:
        """Compute the `Mean-Average-Precision (mAP) and Mean-Average-Recall (mAR)` scores. All detections added in
        the `update()` method are included.

        .. note::
            Scores are calculated with @[ IoU=0.50:0.95 | area=all | maxDets=100 ]

        Returns:
            dict containing

            - map_value: ``torch.Tensor``
            - mar_value: ``torch.Tensor``
            - map_per_class_value: ``List[torch.Tensor]``
            - mar_per_class_value: ``List[torch.Tensor]``
        """
        coco_target, coco_preds = COCO(), COCO()
        coco_target.dataset = self._get_coco_format(self.groundtruth_boxes, self.groundtruth_labels)
        coco_preds.dataset = self._get_coco_format(self.detection_boxes, self.detection_labels, self.detection_scores)

        with _hide_prints():
            coco_target.createIndex()
            coco_preds.createIndex()
            coco_eval = COCOeval(coco_target, coco_preds, "bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            stats = coco_eval.stats

        new_average_precision = torch.cat(
            (
                self.average_precision,
                torch.tensor(
                    [stats[COCO_STATS_MAP_VALUE_INDEX]], dtype=torch.float, device=self.average_precision.device
                ),
            )
        )
        setattr(self, "average_precision", new_average_precision)

        new_average_recall = torch.cat(
            (
                self.average_recall,
                torch.tensor([stats[COCO_STATS_MAR_VALUE_INDEX]], dtype=torch.float, device=self.average_recall.device),
            )
        )
        setattr(self, "average_recall", new_average_recall)

        # if class mode is enabled, evaluate metrics per class
        for class_id in range(self.num_classes):
            coco_eval.params.catIds = [class_id]
            with _hide_prints():
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                stats = coco_eval.stats

            current_value = getattr(self, f"ap_{class_id}")
            class_map_value = torch.cat(
                (
                    current_value,
                    torch.tensor([stats[COCO_STATS_MAP_VALUE_INDEX]], dtype=torch.float, device=current_value.device),
                )
            )
            setattr(self, f"ap_{class_id}", class_map_value)

            current_value = getattr(self, f"ar_{class_id}")
            class_mar_value = torch.cat(
                (
                    current_value,
                    torch.tensor([stats[COCO_STATS_MAR_VALUE_INDEX]], dtype=torch.float, device=current_value.device),
                )
            )
            setattr(self, f"ar_{class_id}", class_mar_value)

        map_per_class_value = [torch.mean(getattr(self, f"ap_{class_id}")) for class_id in range(self.num_classes)]
        mar_per_class_value = [torch.mean(getattr(self, f"ar_{class_id}")) for class_id in range(self.num_classes)]
        metrics = MAPMetricResults(
            map_value=torch.mean(self.average_precision),
            mar_value=torch.mean(self.average_recall),
            map_per_class_value=map_per_class_value,
            mar_per_class_value=mar_per_class_value,
        )
        return metrics

    def _get_coco_format(self, boxes, labels, scores=None) -> Dict:
        """Transforms and returns all cached targets or predictions in COCO format.

        Format is defined at https://cocodataset.org/#format-data
        """

        images = []
        annotations = []
        annotation_id = 1  # has to start with 1, otherwise COCOEval results are wrong

        for image_id, (image_boxes, image_labels) in enumerate(zip(boxes, labels)):
            image_boxes = image_boxes.cpu().tolist()
            image_labels = image_labels.cpu().tolist()

            images.append({"id": image_id})
            for k, (image_box, image_label) in enumerate(zip(image_boxes, image_labels)):
                if len(image_box) != 4:
                    raise ValueError(
                        f"Invalid input box of sample {image_id}, element {k} (expected 4 values, got {len(image_box)})"
                    )

                if type(image_label) != int:
                    raise ValueError(
                        f"Invalid input class of sample {image_id}, element {k}"
                        f" (expected value of type integer, got type {type(image_label)})"
                    )

                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "bbox": image_box,
                    "category_id": image_label,
                    "area": image_box[2] * image_box[3],
                    "iscrowd": 0,
                }
                if scores is not None:
                    score = scores[image_id][k].cpu().tolist()
                    if type(score) != float:
                        raise ValueError(
                            f"Invalid input score of sample {image_id}, element {k}"
                            f" (expected value of type float, got type {type(score)})"
                        )
                    annotation["score"] = score
                annotations.append(annotation)
                annotation_id += 1

        classes = [{"id": i, "name": str(i)} for i in range(self.num_classes)]
        return {"images": images, "annotations": annotations, "categories": classes}
