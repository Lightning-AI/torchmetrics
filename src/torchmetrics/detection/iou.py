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
from typing import Any, Callable, Dict, List, Optional

import torch
from torch import Tensor

from torchmetrics.detection.helpers import _fix_empty_tensors, _input_validator
from torchmetrics.functional.detection.iou import _iou_compute, _iou_update
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.imports import _TORCHVISION_GREATER_EQUAL_0_8

if _TORCHVISION_GREATER_EQUAL_0_8:
    from torchvision.ops import box_convert
else:
    box_convert = None
    __doctest_skip__ = ["IntersectionOverUnion"]


class IntersectionOverUnion(Metric):
    r"""
    Computes Intersection Over Union (IoU)
    Args:
        box_format:
            Input format of given boxes. Supported formats are ``[`xyxy`, `xywh`, `cxcywh`]``.
        iou_thresholds:
            Optional IoU thresholds for evaluation. If set to `None` the threshold is ignored.
        max_detection_thresholds:
            Thresholds on max detections per image. If set to `None` will use thresholds `[1, 10, 100]`.
            Else please provide a list of ints.
        class_metrics:
            Option to enable per-class metrics for IoU. Has a performance impact.
    """
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = True

    update_fn: Callable[[Tensor, Tensor, bool], Tensor] = _iou_update
    compute_fn: Callable[[Tensor], Tensor] = _iou_compute
    type: str = "iou"

    def __init__(
        self,
        box_format: str = "xyxy",
        iou_threshold: Optional[float] = None,
        class_metrics: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not _TORCHVISION_GREATER_EQUAL_0_8:
            raise ModuleNotFoundError(
                "`BoxIntersectionOverUnion` metric requires that `torchvision` version 0.8.0 or newer is installed."
                " Please install with `pip install torchvision>=0.8` or `pip install torchmetrics[detection]`."
            )

        allowed_box_formats = ("xyxy", "xywh", "cxcywh")
        if box_format not in allowed_box_formats:
            raise ValueError(f"Expected argument `box_format` to be one of {allowed_box_formats} but got {box_format}")

        self.box_format = box_format
        self.iou_threshold = iou_threshold
        self.class_metrics = class_metrics
        self.add_state("detections", default=[], dist_reduce_fx=None)
        self.add_state("detection_scores", default=[], dist_reduce_fx=None)
        self.add_state("detection_labels", default=[], dist_reduce_fx=None)
        self.add_state("groundtruths", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_labels", default=[], dist_reduce_fx=None)
        self.add_state("x", default=Tensor([]), dist_reduce_fx="cat")

        rank_zero_warn(
            "Metric `IoU` will save all IoU > threshold values in buffer."
            " For large datasets with many objects, this may lead to large memory footprint."
        )

    def update(self, preds: List[Dict[str, Tensor]], target: List[Dict[str, Tensor]]) -> None:  # type: ignore
        """Add detections and ground truth to the metric.
        Args:
            preds: A list consisting of dictionaries each containing the key-values
            (each dictionary corresponds to a single image):
            - ``boxes``: ``torch.FloatTensor`` of shape
                [num_boxes, 4] containing `num_boxes` detection boxes of the format
                specified in the contructor. By default, this method expects
                [xmin, ymin, xmax, ymax] in absolute image coordinates.
            - ``scores``: ``torch.FloatTensor`` of shape
                [num_boxes] containing detection scores for the boxes.
            - ``labels``: ``torch.IntTensor`` of shape
                [num_boxes] containing 0-indexed detection classes for the boxes.
            target: A list consisting of dictionaries each containing the key-values
            (each dictionary corresponds to a single image):
            - ``boxes``: ``torch.FloatTensor`` of shape
                [num_boxes, 4] containing `num_boxes` ground truth boxes of the format
                specified in the contructor. By default, this method expects
                [xmin, ymin, xmax, ymax] in absolute image coordinates.
            - ``labels``: ``torch.IntTensor`` of shape
                [num_boxes] containing 1-indexed ground truth classes for the boxes.
        Raises:
            ValueError:
                If ``preds`` is not of type List[Dict[str, Tensor]]
            ValueError:
                If ``target`` is not of type List[Dict[str, Tensor]]
            ValueError:
                If ``preds`` and ``target`` are not of the same length
            ValueError:
                If any of ``preds.boxes``, ``preds.scores``
                and ``preds.labels`` are not of the same length
            ValueError:
                If any of ``target.boxes`` and ``target.labels`` are not of the same length
            ValueError:
                If any box is not type float and of length 4
            ValueError:
                If any class is not type int and of length 1
            ValueError:
                If any score is not type float and of length 1
        """
        _input_validator(preds, target)

        for item in preds:
            det_boxes = box_convert(item["boxes"], in_fmt=self.box_format, out_fmt="xyxy")
            det_boxes = _fix_empty_tensors(det_boxes)
            self.detections.append(det_boxes)
            self.detection_labels.append(item["labels"])
            self.detection_scores.append(item["scores"])

        for item in target:
            gt_boxes = box_convert(item["boxes"], in_fmt=self.box_format, out_fmt="xyxy")
            gt_boxes = _fix_empty_tensors(gt_boxes)
            self.groundtruths.append(gt_boxes)
            self.groundtruth_labels.append(item["labels"])

        result = self.update_fn(det_boxes, gt_boxes, self.iou_threshold)
        self.x.append(result)

    def _get_classes(self) -> List:
        """Returns a list of unique classes found in ground truth and detection data."""
        if len(self.detection_labels) > 0 or len(self.groundtruth_labels) > 0:
            return torch.cat(self.detection_labels + self.groundtruth_labels).unique().tolist()
        return []

    def compute(self) -> dict:
        """Computes IoU based on inputs passed in to ``update`` previously."""
        self.classes = self._get_classes()
        results = {"iou": self.compute_fn(self.x)}
        return results
