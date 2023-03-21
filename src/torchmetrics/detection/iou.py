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
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

import torch
from torch import Tensor

from torchmetrics.detection.helpers import _fix_empty_tensors, _input_validator
from torchmetrics.functional.detection.iou import _iou_compute, _iou_update
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat, dim_zero_mean

# from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.imports import _TORCHVISION_GREATER_EQUAL_0_8

if _TORCHVISION_GREATER_EQUAL_0_8:
    from torchvision.ops import box_convert
else:
    box_convert = None


class IntersectionOverUnion(Metric):
    r"""Computes Intersection Over Union (IoU).

    Args:
        box_format:
            Input format of given boxes. Supported formats are ``[`xyxy`, `xywh`, `cxcywh`]``.
        iou_thresholds:
            Optional IoU thresholds for evaluation. If set to `None` the threshold is ignored.
        class_metrics:
            Option to enable per-class metrics for IoU. Has a performance impact.
        respect_labels:
            Replace IoU values with the `invalid_val` if the labels do not match.
        kwargs:
             Additional keyword arguments, see :ref:`Metric kwargs` for more info.
    """
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = True

    detections: List[Tensor]
    detection_scores: List[Tensor]
    detection_labels: List[Tensor]
    groundtruths: List[Tensor]
    groundtruth_labels: List[Tensor]
    results: List[Tensor]
    labels_eq: List[Tensor]
    _iou_update_fn: Callable[[Tensor, Tensor, Optional[float], float], Tensor]
    _iou_compute_fn: Callable[[Tensor, bool], Tensor]
    _iou_type: str = "iou"
    _invalid_val: float = 0.0

    def __init__(
        self,
        box_format: str = "xyxy",
        iou_threshold: Optional[float] = None,
        class_metrics: bool = False,
        respect_labels: bool = True,
        iou_update_fn: Callable[[Tensor, Tensor, Optional[float], float], Tensor] = _iou_update,
        iou_compute_fn: Callable[[Tensor, bool], Tensor] = _iou_compute,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not _TORCHVISION_GREATER_EQUAL_0_8:
            raise ModuleNotFoundError(
                f"Metric `{self._iou_type.upper()}` requires that `torchvision` version 0.8.0 or newer is installed."
                " Please install with `pip install torchvision>=0.8` or `pip install torchmetrics[detection]`."
            )

        self._iou_update_fn = iou_update_fn
        self._iou_compute_fn = iou_compute_fn

        allowed_box_formats = ("xyxy", "xywh", "cxcywh")
        if box_format not in allowed_box_formats:
            raise ValueError(f"Expected argument `box_format` to be one of {allowed_box_formats} but got {box_format}")

        self.box_format = box_format
        self.iou_threshold = iou_threshold

        if not isinstance(class_metrics, bool):
            raise ValueError("Expected argument `class_metrics` to be a boolean")
        self.class_metrics = class_metrics

        if not isinstance(respect_labels, bool):
            raise ValueError("Expected argument `respect_labels` to be a boolean")
        self.respect_labels = respect_labels

        self.add_state("detections", default=[], dist_reduce_fx=None)
        self.add_state("detection_scores", default=[], dist_reduce_fx=None)
        self.add_state("detection_labels", default=[], dist_reduce_fx=None)
        self.add_state("groundtruths", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_labels", default=[], dist_reduce_fx=None)
        self.add_state("results", default=[], dist_reduce_fx=None)
        self.add_state("labels_eq", default=[], dist_reduce_fx=None)

    def update(self, preds: List[Dict[str, Tensor]], target: List[Dict[str, Tensor]]) -> None:
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

        for p, t in zip(preds, target):
            det_boxes = self._get_safe_item_values(p["boxes"])
            self.detections.append(det_boxes)
            self.detection_labels.append(p["labels"])
            self.detection_scores.append(p["scores"])

            gt_boxes = self._get_safe_item_values(t["boxes"])
            self.groundtruths.append(gt_boxes)
            self.groundtruth_labels.append(t["labels"])

            label_eq = torch.equal(p["labels"], t["labels"])
            # Workaround to persist state, which only works with tensors
            self.labels_eq.append(torch.tensor([label_eq], dtype=torch.bool, device=self.device))

            ious = self._iou_update_fn(det_boxes, gt_boxes, self.iou_threshold, self._invalid_val)
            if self.respect_labels and not label_eq:
                label_diff = p["labels"].unsqueeze(0).T - t["labels"].unsqueeze(0)
                labels_not_eq = label_diff != 0.0
                ious[labels_not_eq] = self._invalid_val
            self.results.append(ious.to(dtype=torch.float, device=self.device))

    def _get_safe_item_values(self, boxes: Tensor) -> Tensor:
        boxes = _fix_empty_tensors(boxes)
        if boxes.numel() > 0:
            boxes = box_convert(boxes, in_fmt=self.box_format, out_fmt="xyxy")
        return boxes

    def _get_gt_classes(self) -> List:
        """Returns a list of unique classes found in ground truth and detection data."""
        if len(self.groundtruth_labels) > 0:
            return torch.cat(self.groundtruth_labels).unique().tolist()
        return []

    def compute(self) -> dict:
        """Computes IoU based on inputs passed in to ``update`` previously."""
        aggregated_iou = dim_zero_cat(
            [self._iou_compute_fn(iou, bool(lbl_eq)) for iou, lbl_eq in zip(self.results, self.labels_eq)]
        )
        results: Dict[str, Tensor] = {f"{self._iou_type}": aggregated_iou.mean()}

        if self.class_metrics:
            class_results: Dict[int, List[Tensor]] = defaultdict(list)
            for iou, label in zip(self.results, self.groundtruth_labels):
                for cl in self._get_gt_classes():
                    masked_iou = iou[:, label == cl]
                    if masked_iou.numel() > 0:
                        class_results[cl].append(self._iou_compute_fn(masked_iou, False))

            results.update(
                {f"{self._iou_type}/cl_{cl}": dim_zero_cat(class_results[cl]).mean() for cl in class_results}
            )
        return results
