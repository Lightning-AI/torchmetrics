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
from typing import Any, Dict, List, Literal, Optional

from torch import Tensor

from torchmetrics.detection.helpers import _input_validator
from torchmetrics.functional.detection.map_3d import mean_average_precision_3d
from torchmetrics.metric import Metric

__all__ = ["MeanAveragePrecision3D"]


class MeanAveragePrecision3D(Metric):
    r"""Compute the ``Mean-Average-Precision (mAP) and Mean-Average-Recall (mAR)`` for 3D object detection.

    This is the 3D counterpart of :class:`~torchmetrics.detection.MeanAveragePrecision`. Instead of 2D bounding
    boxes it operates on axis-aligned 3D bounding boxes, e.g. as produced by detectors operating on point-clouds
    or voxel grids. Because axis-aligned 3D boxes are used, this implementation does not depend on
    ``pycocotools``/``faster_coco_eval`` and instead computes the intersection-over-union and average precision
    directly.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~List`): A list of dictionaries, each containing the key-values

        - ``boxes`` (:class:`~torch.Tensor`): float tensor of shape ``(num_boxes, 6)`` containing ``num_boxes``
          detection boxes of the format specified in the constructor. By default, this method expects
          ``(center_x, center_y, center_z, width, height, depth)`` but can be changed using the ``box_format``
          parameter.
        - ``scores`` (:class:`~torch.Tensor`): float tensor of shape ``(num_boxes)`` containing detection scores
          for the boxes.
        - ``labels`` (:class:`~torch.Tensor`): integer tensor of shape ``(num_boxes)`` containing 0-indexed
          detection classes for the boxes.

    - ``target`` (:class:`~List`): A list of dictionaries, each containing the key-values

        - ``boxes`` (:class:`~torch.Tensor`): float tensor of shape ``(num_boxes, 6)`` containing ``num_boxes``
          ground truth boxes of the format specified in the constructor.
        - ``labels`` (:class:`~torch.Tensor`): integer tensor of shape ``(num_boxes)`` containing 0-indexed ground
          truth classes for the boxes.

    As output of ``forward`` and ``compute`` the metric returns a dictionary containing ``map``, ``map_50``,
    ``map_75``, ``mar_{max_det}`` (one key per value in ``max_detection_thresholds``), ``map_per_class``,
    ``mar_{max_det}_per_class`` and ``classes``. See
    :func:`~torchmetrics.functional.detection.mean_average_precision_3d` for a description of these values.

    Args:
        box_format: Format of the input 3D boxes. Either ``"xyzwhd"`` (center x, y, z and width, height, depth) or
            ``"xyzxyz"`` (min x, y, z and max x, y, z corners).
        iou_thresholds: List of IoU thresholds (default is ``[0.5, 0.55, ..., 0.95]``).
        rec_thresholds: Unused, kept for API parity with the 2D metric.
        max_detection_thresholds: List of maximum detections per sample (default is ``[1, 10, 100]``).
        class_metrics: Whether to compute per-class mAP and mAR metrics.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example::

        >>> from torch import tensor
        >>> from torchmetrics.detection import MeanAveragePrecision3D
        >>> preds = [
        ...   dict(
        ...     boxes=tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]]),
        ...     scores=tensor([0.9]),
        ...     labels=tensor([0]),
        ...   )
        ... ]
        >>> target = [
        ...   dict(
        ...     boxes=tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]]),
        ...     labels=tensor([0]),
        ...   )
        ... ]
        >>> metric = MeanAveragePrecision3D()
        >>> metric.update(preds, target)
        >>> result = metric.compute()
        >>> print(f"mAP: {result['map']:.4f}, mAP@0.5: {result['map_50']:.4f}")
        mAP: 1.0000, mAP@0.5: 1.0000

    """

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = True
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    detection_box: List[Tensor]
    detection_scores: List[Tensor]
    detection_labels: List[Tensor]
    groundtruth_box: List[Tensor]
    groundtruth_labels: List[Tensor]

    def __init__(
        self,
        box_format: Literal["xyzwhd", "xyzxyz"] = "xyzwhd",
        iou_thresholds: Optional[List[float]] = None,
        rec_thresholds: Optional[List[float]] = None,
        max_detection_thresholds: Optional[List[int]] = None,
        class_metrics: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        allowed_box_formats = ("xyzwhd", "xyzxyz")
        if box_format not in allowed_box_formats:
            raise ValueError(f"Expected argument `box_format` to be one of {allowed_box_formats} but got {box_format}")
        self.box_format = box_format

        if iou_thresholds is not None and not isinstance(iou_thresholds, list):
            raise ValueError(
                f"Expected argument `iou_thresholds` to either be `None` or a list of floats but got {iou_thresholds}"
            )
        self.iou_thresholds = iou_thresholds

        if rec_thresholds is not None and not isinstance(rec_thresholds, list):
            raise ValueError(
                f"Expected argument `rec_thresholds` to either be `None` or a list of floats but got {rec_thresholds}"
            )
        self.rec_thresholds = rec_thresholds

        if max_detection_thresholds is not None and not isinstance(max_detection_thresholds, list):
            raise ValueError(
                f"Expected argument `max_detection_thresholds` to either be `None` or a list of ints"
                f" but got {max_detection_thresholds}"
            )
        self.max_detection_thresholds = max_detection_thresholds

        if not isinstance(class_metrics, bool):
            raise ValueError("Expected argument `class_metrics` to be a boolean")
        self.class_metrics = class_metrics

        self.add_state("detection_box", default=[], dist_reduce_fx=None)
        self.add_state("detection_scores", default=[], dist_reduce_fx=None)
        self.add_state("detection_labels", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_box", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_labels", default=[], dist_reduce_fx=None)

    def update(self, preds: List[Dict[str, Tensor]], target: List[Dict[str, Tensor]]) -> None:
        """Update metric state."""
        _input_validator(preds, target, iou_type="bbox")
        for item in preds:
            self.detection_box.append(item["boxes"])
            self.detection_scores.append(item["scores"])
            self.detection_labels.append(item["labels"])
        for item in target:
            self.groundtruth_box.append(item["boxes"])
            self.groundtruth_labels.append(item["labels"])

    def compute(self) -> Dict[str, Tensor]:
        """Computes the metric."""
        preds = [
            {"boxes": box, "scores": scores, "labels": labels}
            for box, scores, labels in zip(self.detection_box, self.detection_scores, self.detection_labels)
        ]
        target = [
            {"boxes": box, "labels": labels} for box, labels in zip(self.groundtruth_box, self.groundtruth_labels)
        ]
        return mean_average_precision_3d(
            preds,
            target,
            box_format=self.box_format,
            iou_thresholds=self.iou_thresholds,
            rec_thresholds=self.rec_thresholds,
            max_detection_thresholds=self.max_detection_thresholds,
            class_metrics=self.class_metrics,
        )
