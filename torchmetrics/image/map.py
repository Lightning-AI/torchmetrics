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
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
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


class GroundtruthDict(Dict):
    groundtruth_boxes: Union[torch.FloatTensor, np.ndarray, List[float]]
    groundtruth_classes: Union[torch.IntTensor, np.ndarray, List[int]]


class DetectionsDict(Dict):
    detection_boxes: Union[torch.FloatTensor, np.ndarray, List[float]]
    detection_scores: Union[torch.FloatTensor, np.ndarray, List[float]]
    detection_classes: Union[torch.IntTensor, np.ndarray, List[int]]


@dataclass
class MAPMetricResults:
    """Dataclass to wrap the final mAP results."""

    map_value: Tensor
    mar_value: Tensor
    map_per_class_value: List[Tensor]
    mar_per_class_value: List[Tensor]


class _hide_prints:
    """Internal helper context to suppress the default output of the pycocotools package."""

    def __enter__(self) -> None:
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore
        sys.stdout.close()
        sys.stdout = self._original_stdout


def _input_validator(preds: List[DetectionsDict], target: List[GroundtruthDict]) -> None:
    if not isinstance(preds, list):
        raise ValueError("Expected argument `preds` to be of type List")
    if not isinstance(target, list):
        raise ValueError("Expected argument `target` to be of type List")
    if len(preds) != len(target):
        raise ValueError("Expected argument `preds` and `target` to have the same length")

    for k in ["detection_boxes", "detection_scores", "detection_classes"]:
        if any(k not in p for p in preds):
            raise ValueError(f"Expected all dicts in `preds` to contain the `{k}` key")

    for k in ["groundtruth_boxes", "groundtruth_classes"]:
        if any(k not in p for p in target):
            raise ValueError(f"Expected all dicts in `target` to contain the `{k}` key")

    # TODO: add more checking


class MAP(Metric):
    r"""
    Computes the `Mean-Average-Precision (mAP) and Mean-Average-Recall (mAR)`_ for object detection predictions.
    Optionally, the mAP and mAR values can be calculated per class.

    Predicted boxes and targets have to be in COCO format with the box score at the end
    (x-top left, y-top left, width, height, score). See the `update` function for more information
    about the input format to this metric.

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
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)

    Raises:
        RuntimeError:
            If ``pycocotools`` is not installed
        ValueError:
            If ``num_classes`` is not an integer larger or equal to 0
    """

    def __init__(
        self,
        num_classes: int = 0,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )

        super().__init__(dist_sync_on_step=dist_sync_on_step)

        if not _PYCOCOTOOLS_AVAILABLE:
            raise RuntimeError(
                "MAP metric requires pycocotools are installed"
                "Please install as `pip install pycocotools` or `pip install torchmetrics[image]`"
            )
        self.add_state("average_precision", default=torch.tensor(data=[], dtype=torch.float), dist_reduce_fx="mean")
        self.add_state("average_recall", default=torch.tensor(data=[], dtype=torch.float), dist_reduce_fx="mean")
        if not isinstance(num_classes, int) and num_classes < 0:
            raise ValueError("Expected argument `num_classes` to be a integer larger or equal to 0")
        self.num_classes = num_classes

        for class_id in range(num_classes):
            self.add_state(f"ap_{class_id}", default=torch.tensor(data=[], dtype=torch.float), dist_reduce_fx="mean")
            self.add_state(f"ar_{class_id}", default=torch.tensor(data=[], dtype=torch.float), dist_reduce_fx="mean")

    def update(self, preds: List[DetectionsDict], target: List[GroundtruthDict]) -> None:  # type: ignore
        """Updates mAP and mAR values with metric values from given predictions and groundtruth.

        Args:
            preds:
                A list consisting of dictionaries each containing the key-values
                (each dictionary corresponds to a single image):

                - ``detection_boxes``: torch.FloatTensor or float32 numpy array of shape
                    [num_boxes, 4] containing `num_boxes` detection boxes of the format
                    [ymin, xmin, ymax, xmax] in absolute image coordinates.
                - ``detection_scores``: torch.FloatTensor or float32 numpy array of shape
                    [num_boxes] containing detection scores for the boxes.
                - ``detection_classes``: torch.IntTensor or integer numpy array of shape
                    [num_boxes] containing 0-indexed detection classes for the boxes.

            target:
                A list consisting of dictionaries each containing the key-values
                (each dictionary corresponds to a single image):

                - ``groundtruth_boxes``: torch.FloatTensor or float32 numpy array of shape
                    [num_boxes, 4] containing `num_boxes` groundtruth boxes of the format
                    [ymin, xmin, ymax, xmax] in absolute image coordinates.
                - ``groundtruth_classes``: integer numpy array of shape
                    [num_boxes] containing 1-indexed groundtruth classes for the boxes.

        Raises:
            ValueError:
                If ``preds`` is not of type List[DetectionsDict]
            ValueError:
                If ``target`` is not of type List[GroundtruthDict]
            ValueError:
                If `preds` and `target` are not of the same length
            ValueError:
                If any of `preds.detection_boxes`, `preds.detection_scores`
                and `preds.detection_classes` are not of the same length
            ValueError:
                If any of `target.groundtruth_boxes` and `target.groundtruth_classes` are not of the same length
            ValueError:
                If any box is not type float and of length 4
            ValueError:
                If any class is not type int and of length 1
            ValueError:
                If any score is not type float and of length 1
        """
        _input_validator(preds, target)

        coco_target, coco_preds = COCO(), COCO()

        coco_target.dataset = self._get_coco_format(inputs=target)
        coco_preds.dataset = self._get_coco_format(inputs=preds, is_pred=True)

        with _hide_prints():
            coco_target.createIndex()
            coco_preds.createIndex()
            coco_eval = COCOeval(coco_target, coco_preds, "bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            stats = coco_eval.stats

        self.average_precision: Tensor = torch.cat(
            (
                self.average_precision,
                torch.tensor(
                    [stats[COCO_STATS_MAP_VALUE_INDEX]], dtype=torch.float, device=self.average_precision.device
                ),
            )
        )

        self.average_recall: Tensor = torch.cat(
            (
                self.average_recall,
                torch.tensor([stats[COCO_STATS_MAR_VALUE_INDEX]], dtype=torch.float, device=self.average_recall.device),
            )
        )

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
            class_map_value = torch.cat(
                (
                    current_value,
                    torch.tensor([stats[COCO_STATS_MAR_VALUE_INDEX]], dtype=torch.float, device=current_value.device),
                )
            )
            setattr(self, f"ar_{class_id}", class_map_value)

    def compute(self) -> MAPMetricResults:
        map_per_class_value = [torch.mean(getattr(self, f"ap_{class_id}")) for class_id in range(self.num_classes)]
        mar_per_class_value = [torch.mean(getattr(self, f"ar_{class_id}")) for class_id in range(self.num_classes)]
        metrics = MAPMetricResults(
            map_value=torch.mean(self.average_precision),
            mar_value=torch.mean(self.average_recall),
            map_per_class_value=map_per_class_value,
            mar_per_class_value=mar_per_class_value,
        )
        return metrics

    def _get_coco_format(
        self, inputs: Union[List[GroundtruthDict], List[DetectionsDict]], is_pred: bool = False
    ) -> Dict:
        images = []
        annotations = []
        annotation_id = 1  # has to start with 1, otherwise COCOEval results are wrong

        for i, input in enumerate(inputs):
            images.append({"id": i})
            boxes, classes, scores = self._get_values_for_sample(input, is_pred)

            if len(boxes) != len(classes):
                raise ValueError(
                    f"Input boxes and classes of sample {i} have a"
                    f" different length (expected {len(boxes)} classes, got {len(classes)}"
                )
            if is_pred and scores is not None and len(boxes) != len(scores):
                raise ValueError(
                    f"Input boxes and scores of sample {i} have a different"
                    f" length (expected {len(boxes)} scores, got {len(scores)}"
                )

            for k, (box, label) in enumerate(zip(boxes, classes)):
                if len(box) != 4:
                    raise ValueError(f"Invalid input box of sample {i}, element {k} (expected 4 values, got {len(box)}")
                if type(label) != int:
                    raise ValueError(
                        f"Invalid input class of sample {i}, element {k}"
                        f" (expected value of type integer, got type {type(label)})"
                    )
                annotation = {
                    "id": annotation_id,
                    "image_id": i,
                    "bbox": box,
                    "category_id": label,
                    "area": box[2] * box[3],
                    "iscrowd": 0,
                }
                if is_pred and scores is not None:
                    score = scores[k]
                    if type(score) != float:
                        raise ValueError(
                            f"Invalid input score of sample {i}, element {k}"
                            f" (expected value of type float, got type {type(score)})"
                        )
                    annotation["score"] = score
                annotations.append(annotation)
                annotation_id += 1

        classes = [{"id": i, "name": str(i)} for i in range(self.num_classes)]
        return {"images": images, "annotations": annotations, "categories": classes}

    def _get_values_for_sample(
        self, input: Union[GroundtruthDict, DetectionsDict], is_pred: bool
    ) -> Tuple[List, List, Optional[List]]:
        boxes = input["detection_boxes"] if is_pred else input["groundtruth_boxes"]
        classes = input["detection_classes"] if is_pred else input["groundtruth_classes"]

        scores_list: Optional[List] = None
        if type(boxes) is torch.Tensor:
            boxes_list = boxes.cpu().tolist()
        elif type(boxes) is np.ndarray:
            boxes_list = boxes.tolist()
        else:
            boxes_list = boxes
        boxes_list = [
            box.cpu().tolist() if type(box) == torch.Tensor else box.tolist() if type(box) is np.ndarray else box
            for box in boxes_list
        ]

        if type(boxes) is torch.Tensor:
            classes_list = classes.cpu().tolist()
        elif type(boxes) is np.ndarray:
            classes_list = classes.tolist()
        else:
            classes_list = classes
        classes_list = [
            label.cpu().tolist()
            if type(label) == torch.Tensor
            else label.tolist()
            if type(label) is np.ndarray
            else label
            for label in classes_list
        ]

        if is_pred:
            scores = input["detection_scores"]
            if type(scores) is torch.Tensor:
                scores_list = scores.cpu().tolist()
            elif type(scores) is np.ndarray:
                scores_list = scores.tolist()
            else:
                scores_list = scores
            scores_list = [
                score.cpu().tolist()
                if type(score) == torch.Tensor
                else score.tolist()
                if type(score) is np.ndarray
                else score
                for score in scores_list
            ]

        return boxes_list, classes_list, scores_list
