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
import logging
import sys
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
from torch import Tensor
from torch import distributed as dist
from typing_extensions import Literal

from torchmetrics.detection.helpers import _fix_empty_tensors, _input_validator
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import (
    _MATPLOTLIB_AVAILABLE,
    _PYCOCOTOOLS_AVAILABLE,
    _TORCHVISION_GREATER_EQUAL_0_8,
)
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["MeanAveragePrecision.plot"]


if _TORCHVISION_GREATER_EQUAL_0_8:
    from torchvision.ops import box_convert
else:
    box_convert = None
    __doctest_skip__ = ["MeanAveragePrecision.plot", "MeanAveragePrecision"]


if _PYCOCOTOOLS_AVAILABLE:
    import pycocotools.mask as mask_utils
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
else:
    COCO, COCOeval = None, None
    mask_utils = None
    __doctest_skip__ = ["MeanAveragePrecision.plot", "MeanAveragePrecision"]


log = logging.getLogger(__name__)


@dataclass
class MAPMetricResults:
    """Dataclass to wrap the final mAP results."""

    map: Tensor  # noqa: A003
    map_50: Tensor
    map_75: Tensor
    map_small: Tensor
    map_medium: Tensor
    map_large: Tensor
    mar_1: Tensor
    mar_10: Tensor
    mar_100: Tensor
    mar_small: Tensor
    mar_medium: Tensor
    mar_large: Tensor
    map_per_class: Tensor
    mar_100_per_class: Tensor
    classes: Tensor

    def __getitem__(self, key: str) -> Union[Tensor, List[Tensor]]:
        """Enables accessing the results via `result['map']` instead of `result.map`."""
        return getattr(self, key)


class WriteToLog:
    """Logging class to move logs to log.debug()."""

    def write(self, buf: str) -> None:
        """Write to log.debug() instead of stdout."""
        for line in buf.rstrip().splitlines():
            log.debug(line.rstrip())

    def flush(self) -> None:
        """Flush the logger."""
        for handler in log.handlers:
            handler.flush()

    def close(self) -> None:
        """Close the logger."""
        for handler in log.handlers:
            handler.close()


class _HidePrints:
    """Internal helper context to suppress the default output of the pycocotools package."""

    def __init__(self) -> None:
        """Initialize the context."""
        self._original_stdout = None

    def __enter__(self) -> None:
        """Redirect stdout to log.debug()."""
        self._original_stdout = sys.stdout  # type: ignore
        sys.stdout = WriteToLog()  # type: ignore

    def __exit__(
        self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_t: Optional[TracebackType]
    ) -> None:  # type: ignore
        """Restore stdout."""
        sys.stdout.close()
        sys.stdout = self._original_stdout  # type: ignore


class MeanAveragePrecision(Metric):
    r"""Compute the `Mean-Average-Precision (mAP) and Mean-Average-Recall (mAR)`_ for object detection predictions.

    .. math::
        \text{mAP} = \frac{1}{n} \sum_{i=1}^{n} AP_i

    where :math:`AP_i` is the average precision for class :math:`i` and :math:`n` is the number of classes. The average
    precision is defined as the area under the precision-recall curve. If argument `class_metrics` is set to ``True``,
    the metric will also return the mAP/mAR per class.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~List`): A list consisting of dictionaries each containing the key-values
      (each dictionary corresponds to a single image). Parameters that should be provided per dict

        - boxes: (:class:`~torch.FloatTensor`) of shape ``(num_boxes, 4)`` containing ``num_boxes`` detection
          boxes of the format specified in the constructor.
          By default, this method expects ``(xmin, ymin, xmax, ymax)`` in absolute image coordinates.
        - scores: :class:`~torch.FloatTensor` of shape ``(num_boxes)`` containing detection scores for the boxes.
        - labels: :class:`~torch.IntTensor` of shape ``(num_boxes)`` containing 0-indexed detection classes for
          the boxes.
        - masks: :class:`~torch.bool` of shape ``(num_boxes, image_height, image_width)`` containing boolean masks.
          Only required when `iou_type="segm"`.

    - ``target`` (:class:`~List`) A list consisting of dictionaries each containing the key-values
      (each dictionary corresponds to a single image). Parameters that should be provided per dict:

        - boxes: :class:`~torch.FloatTensor` of shape ``(num_boxes, 4)`` containing ``num_boxes`` ground truth
          boxes of the format specified in the constructor.
          By default, this method expects ``(xmin, ymin, xmax, ymax)`` in absolute image coordinates.
        - labels: :class:`~torch.IntTensor` of shape ``(num_boxes)`` containing 0-indexed ground truth
          classes for the boxes.
        - masks: :class:`~torch.bool` of shape ``(num_boxes, image_height, image_width)`` containing boolean masks.
          Only required when `iou_type="segm"`.
        - iscrowd: :class:`~torch.IntTensor` of shape ``(num_boxes)`` containing 0/1 values indicating whether
            the bounding box/masks indicate a crowd of objects. Value is optional, and if not provided it will
            automatically be set to 0.
        - area: :class:`~torch.FloatTensor` of shape ``(num_boxes)`` containing the area of the object. Value if
            optional, and if not provided will be automatically calculated based on the bounding box/masks provided.
            Only affects which samples contribute to the `map_small`, `map_medium`, `map_large` values

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``map_dict``: A dictionary containing the following key-values:

        - map: (:class:`~torch.Tensor`)
        - map_small: (:class:`~torch.Tensor`)
        - map_medium:(:class:`~torch.Tensor`)
        - map_large: (:class:`~torch.Tensor`)
        - mar_1: (:class:`~torch.Tensor`)
        - mar_10: (:class:`~torch.Tensor`)
        - mar_100: (:class:`~torch.Tensor`)
        - mar_small: (:class:`~torch.Tensor`)
        - mar_medium: (:class:`~torch.Tensor`)
        - mar_large: (:class:`~torch.Tensor`)
        - map_50: (:class:`~torch.Tensor`) (-1 if 0.5 not in the list of iou thresholds)
        - map_75: (:class:`~torch.Tensor`) (-1 if 0.75 not in the list of iou thresholds)
        - map_per_class: (:class:`~torch.Tensor`) (-1 if class metrics are disabled)
        - mar_100_per_class: (:class:`~torch.Tensor`) (-1 if class metrics are disabled)
        - classes (:class:`~torch.Tensor`)

    For an example on how to use this metric check the `torchmetrics mAP example`_.

    .. note::
        ``map`` score is calculated with @[ IoU=self.iou_thresholds | area=all | max_dets=max_detection_thresholds ].
        Caution: If the initialization parameters are changed, dictionary keys for mAR can change as well.
        The default properties are also accessible via fields and will raise an ``AttributeError`` if not available.

    .. note::
        This metric is following the mAP implementation of
        `pycocotools <https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools>`_,
        a standard implementation for the mAP metric for object detection.

    .. note::
        This metric requires you to have `torchvision` version 0.8.0 or newer installed
        (with corresponding version 1.7.0 of torch or newer). This metric requires `pycocotools`
        installed when iou_type is `segm`. Please install with ``pip install torchvision`` or
        ``pip install torchmetrics[detection]``.

    Args:
        box_format:
            Input format of given boxes. Supported formats are ``[`xyxy`, `xywh`, `cxcywh`]``.
        iou_type:
            Type of input (either masks or bounding-boxes) used for computing IOU.
            Supported IOU types are ``["bbox", "segm"]``.
            If using ``"segm"``, masks should be provided (see :meth:`update`).
        iou_thresholds:
            IoU thresholds for evaluation. If set to ``None`` it corresponds to the stepped range ``[0.5,...,0.95]``
            with step ``0.05``. Else provide a list of floats.
        rec_thresholds:
            Recall thresholds for evaluation. If set to ``None`` it corresponds to the stepped range ``[0,...,1]``
            with step ``0.01``. Else provide a list of floats.
        max_detection_thresholds:
            Thresholds on max detections per image. If set to `None` will use thresholds ``[1, 10, 100]``.
            Else, please provide a list of ints.
        class_metrics:
            Option to enable per-class metrics for mAP and mAR_100. Has a performance impact.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ModuleNotFoundError:
            If ``torchvision`` is not installed or version installed is lower than 0.8.0
        ModuleNotFoundError:
            If ``iou_type`` is equal to ``segm`` and ``pycocotools`` is not installed
        ValueError:
            If ``class_metrics`` is not a boolean
        ValueError:
            If ``preds`` is not of type (:class:`~List[Dict[str, Tensor]]`)
        ValueError:
            If ``target`` is not of type ``List[Dict[str, Tensor]]``
        ValueError:
            If ``preds`` and ``target`` are not of the same length
        ValueError:
            If any of ``preds.boxes``, ``preds.scores`` and ``preds.labels`` are not of the same length
        ValueError:
            If any of ``target.boxes`` and ``target.labels`` are not of the same length
        ValueError:
            If any box is not type float and of length 4
        ValueError:
            If any class is not type int and of length 1
        ValueError:
            If any score is not type float and of length 1

    Example:
        >>> from torch import tensor
        >>> from torchmetrics.detection import MeanAveragePrecision
        >>> preds = [
        ...   dict(
        ...     boxes=tensor([[258.0, 41.0, 606.0, 285.0]]),
        ...     scores=tensor([0.536]),
        ...     labels=tensor([0]),
        ...   )
        ... ]
        >>> target = [
        ...   dict(
        ...     boxes=tensor([[214.0, 41.0, 562.0, 285.0]]),
        ...     labels=tensor([0]),
        ...   )
        ... ]
        >>> metric = MeanAveragePrecision()
        >>> metric.update(preds, target)
        >>> from pprint import pprint
        >>> pprint(metric.compute())
        {'classes': tensor(0, dtype=torch.int32),
         'map': tensor(0.6000),
         'map_50': tensor(1.),
         'map_75': tensor(1.),
         'map_large': tensor(0.6000),
         'map_medium': tensor(-1.),
         'map_per_class': tensor(-1.),
         'map_small': tensor(-1.),
         'mar_1': tensor(0.6000),
         'mar_10': tensor(0.6000),
         'mar_100': tensor(0.6000),
         'mar_100_per_class': tensor(-1.),
         'mar_large': tensor(0.6000),
         'mar_medium': tensor(-1.),
         'mar_small': tensor(-1.)}
    """
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = True
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    detections: List[Tensor]
    detection_scores: List[Tensor]
    detection_labels: List[Tensor]
    groundtruths: List[Tensor]
    groundtruth_labels: List[Tensor]
    groundtruth_crowds: List[Tensor]
    groundtruth_area: List[Tensor]

    def __init__(
        self,
        box_format: str = "xyxy",
        iou_type: Literal["bbox", "segm"] = "bbox",
        iou_thresholds: Optional[List[float]] = None,
        rec_thresholds: Optional[List[float]] = None,
        max_detection_thresholds: Optional[List[int]] = None,
        class_metrics: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not _PYCOCOTOOLS_AVAILABLE:
            raise ImportError(
                "`MAP` metric requires that `pycocotools` installed."
                " Please install with `pip install pycocotools` or `pip install torchmetrics[detection]`"
            )
        if not _TORCHVISION_GREATER_EQUAL_0_8:
            raise ModuleNotFoundError(
                "`MeanAveragePrecision` metric requires that `torchvision` version 0.8.0 or newer is installed."
                " Please install with `pip install torchvision>=0.8` or `pip install torchmetrics[detection]`."
            )

        allowed_box_formats = ("xyxy", "xywh", "cxcywh")
        if box_format not in allowed_box_formats:
            raise ValueError(f"Expected argument `box_format` to be one of {allowed_box_formats} but got {box_format}")
        self.box_format = box_format

        allowed_iou_types = ("segm", "bbox")
        if iou_type not in allowed_iou_types:
            raise ValueError(f"Expected argument `iou_type` to be one of {allowed_iou_types} but got {iou_type}")
        self.iou_type = iou_type

        if iou_thresholds is not None and not isinstance(iou_thresholds, list):
            raise ValueError(
                f"Expected argument `iou_thresholds` to either be `None` or a list of floats but got {iou_thresholds}"
            )
        self.iou_thresholds = iou_thresholds or torch.linspace(0.5, 0.95, round((0.95 - 0.5) / 0.05) + 1).tolist()

        if rec_thresholds is not None and not isinstance(rec_thresholds, list):
            raise ValueError(
                f"Expected argument `rec_thresholds` to either be `None` or a list of floats but got {rec_thresholds}"
            )
        self.rec_thresholds = rec_thresholds or torch.linspace(0.0, 1.00, round(1.00 / 0.01) + 1).tolist()

        if max_detection_thresholds is not None and not isinstance(max_detection_thresholds, list):
            raise ValueError(
                f"Expected argument `max_detection_thresholds` to either be `None` or a list of ints"
                f" but got {max_detection_thresholds}"
            )
        max_det_thr, _ = torch.sort(torch.tensor(max_detection_thresholds or [1, 10, 100], dtype=torch.int))
        self.max_detection_thresholds = max_det_thr.tolist()

        if not isinstance(class_metrics, bool):
            raise ValueError("Expected argument `class_metrics` to be a boolean")
        self.class_metrics = class_metrics

        self.add_state("detections", default=[], dist_reduce_fx=None)
        self.add_state("detection_scores", default=[], dist_reduce_fx=None)
        self.add_state("detection_labels", default=[], dist_reduce_fx=None)
        self.add_state("groundtruths", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_labels", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_crowds", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_area", default=[], dist_reduce_fx=None)

    def update(self, preds: List[Dict[str, Tensor]], target: List[Dict[str, Tensor]]) -> None:  # type: ignore
        """Update metric state."""
        _input_validator(preds, target, iou_type=self.iou_type)

        for item in preds:
            detections = self._get_safe_item_values(item)

            self.detections.append(detections)
            self.detection_labels.append(item["labels"])
            self.detection_scores.append(item["scores"])

        for item in target:
            groundtruths = self._get_safe_item_values(item)
            self.groundtruths.append(groundtruths)
            self.groundtruth_labels.append(item["labels"])
            self.groundtruth_crowds.append(item.get("iscrowd", torch.zeros_like(item["labels"])))
            self.groundtruth_area.append(item.get("area", -1 * torch.zeros_like(item["labels"])))

    def compute(self) -> dict:
        """Computes the metric."""
        coco_target, coco_preds = COCO(), COCO()

        coco_target.dataset = self._get_coco_format(
            self.groundtruths, self.groundtruth_labels, crowds=self.groundtruth_crowds, area=self.groundtruth_area
        )
        coco_preds.dataset = self._get_coco_format(self.detections, self.detection_labels, scores=self.detection_scores)

        with _HidePrints():
            coco_target.createIndex()
            coco_preds.createIndex()

            coco_eval = COCOeval(coco_target, coco_preds, iouType=self.iou_type)
            coco_eval.params.iouThrs = np.array(self.iou_thresholds, dtype=np.float64)
            coco_eval.params.recThrs = np.array(self.rec_thresholds, dtype=np.float64)
            coco_eval.params.maxDets = self.max_detection_thresholds

            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            stats = coco_eval.stats

        # if class mode is enabled, evaluate metrics per class
        if self.class_metrics:
            map_per_class_list = []
            mar_100_per_class_list = []
            for class_id in torch.cat(self.detection_labels + self.groundtruth_labels).unique().cpu().tolist():
                coco_eval.params.catIds = [class_id]
                with _HidePrints():
                    coco_eval.evaluate()
                    coco_eval.accumulate()
                    coco_eval.summarize()
                    class_stats = coco_eval.stats

                map_per_class_list.append(torch.tensor([class_stats[0]]))
                mar_100_per_class_list.append(torch.tensor([class_stats[8]]))

            map_per_class_values = torch.tensor(map_per_class_list, dtype=torch.float32)
            mar_100_per_class_values = torch.tensor(mar_100_per_class_list, dtype=torch.float32)
        else:
            map_per_class_values: Tensor = torch.tensor([-1], dtype=torch.float32)
            mar_100_per_class_values: Tensor = torch.tensor([-1], dtype=torch.float32)

        metrics = MAPMetricResults(
            map=torch.tensor([stats[0]], dtype=torch.float32),
            map_50=torch.tensor([stats[1]], dtype=torch.float32),
            map_75=torch.tensor([stats[2]], dtype=torch.float32),
            map_small=torch.tensor([stats[3]], dtype=torch.float32),
            map_medium=torch.tensor([stats[4]], dtype=torch.float32),
            map_large=torch.tensor([stats[5]], dtype=torch.float32),
            mar_1=torch.tensor([stats[6]], dtype=torch.float32),
            mar_10=torch.tensor([stats[7]], dtype=torch.float32),
            mar_100=torch.tensor([stats[8]], dtype=torch.float32),
            mar_small=torch.tensor([stats[9]], dtype=torch.float32),
            mar_medium=torch.tensor([stats[10]], dtype=torch.float32),
            mar_large=torch.tensor([stats[11]], dtype=torch.float32),
            map_per_class=map_per_class_values,
            mar_100_per_class=mar_100_per_class_values,
            classes=torch.tensor(self._get_classes(), dtype=torch.int32),
        )

        return metrics.__dict__

    @staticmethod
    def coco_to_tm(
        coco_preds: str,
        coco_target: str,
        iou_type: Literal["bbox", "segm"] = "bbox",
    ) -> Tuple[List[Dict[str, Tensor]], List[Dict[str, Tensor]]]:
        """Convert coco format to the input format of the map metric.

        Args:
            coco_preds: Path to the json file containing the predictions in coco format
            coco_target: Path to the json file containing the targets in coco format
            iou_type: Type of input, either `bbox` for bounding boxes or `segm` for segmentation masks

        Returns:
            preds: List of dictionaries containing the predictions in the input format of this metric
            target: List of dictionaries containing the targets in the input format of this metric

        """
        gt = COCO(coco_target)
        dt = gt.loadRes(coco_preds)

        gt_dataset = gt.dataset["annotations"]
        dt_dataset = dt.dataset["annotations"]

        target = {}
        for t in gt_dataset:
            if t["image_id"] not in target:
                target[t["image_id"]] = {
                    "boxes" if iou_type == "bbox" else "masks": [],
                    "labels": [],
                    "iscrowd": [],
                    "area": [],
                }
            if iou_type == "bbox":
                target[t["image_id"]]["boxes"].append(t["bbox"])
            else:
                target[t["image_id"]]["masks"].append(gt.annToMask(t))
            target[t["image_id"]]["labels"].append(t["category_id"])
            target[t["image_id"]]["iscrowd"].append(t["iscrowd"])
            target[t["image_id"]]["area"].append(t["area"])

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
        for k in target:  # add empty predictions for images without predictions
            if k not in preds:
                preds[k] = {"boxes" if iou_type == "bbox" else "masks": [], "scores": [], "labels": []}

        batched_preds, batched_target = [], []
        for key in target:
            name = "boxes" if iou_type == "bbox" else "masks"
            batched_preds.append(
                {
                    name: torch.tensor(preds[key]["boxes"])
                    if iou_type == "bbox"
                    else torch.tensor(preds[key]["masks"]),
                    "scores": torch.tensor(preds[key]["scores"]),
                    "labels": torch.tensor(preds[key]["labels"]),
                }
            )
            batched_target.append(
                {
                    name: torch.tensor(target[key]["boxes"])
                    if iou_type == "bbox"
                    else torch.tensor(target[key]["masks"]),
                    "labels": torch.tensor(target[key]["labels"]),
                    "iscrowd": torch.tensor(target[key]["iscrowd"]),
                    "area": torch.tensor(target[key]["area"]),
                }
            )

        return batched_preds, batched_target

    def tm_to_coco(self, name: str = "tm_map_input") -> None:
        """Write the input to the map metric to a json file in coco format.

        Args:
            name: Name of the output file, which will be appended with "_preds.json" and "_target.json"
        """
        target_dataset = self._get_coco_format(self.groundtruths, self.groundtruth_labels)
        preds_dataset = self._get_coco_format(self.detections, self.detection_labels, self.detection_scores)

        preds_json = json.dumps(preds_dataset["annotations"], indent=4)
        target_json = json.dumps(target_dataset, indent=4)

        with open(f"{name}_preds.json", "w") as f:
            f.write(preds_json)

        with open(f"{name}_target.json", "w") as f:
            f.write(target_json)

    def _get_safe_item_values(self, item: Dict[str, Any]) -> Union[Tensor, Tuple]:
        """Convert and return the boxes or masks from the item depending on the iou_type.

        Args:
            item: input dictionary containing the boxes or masks

        Returns:
            boxes or masks depending on the iou_type

        """
        if self.iou_type == "bbox":
            boxes = _fix_empty_tensors(item["boxes"])
            if boxes.numel() > 0:
                boxes = box_convert(boxes, in_fmt=self.box_format, out_fmt="xyxy")
            return boxes
        if self.iou_type == "segm":
            masks = []
            for i in item["masks"].cpu().numpy():
                rle = mask_utils.encode(np.asfortranarray(i))
                masks.append((tuple(rle["size"]), rle["counts"]))
            return tuple(masks)
        raise Exception(f"IOU type {self.iou_type} is not supported")

    def _get_classes(self) -> List:
        """Return a list of unique classes found in ground truth and detection data."""
        if len(self.detection_labels) > 0 or len(self.groundtruth_labels) > 0:
            return torch.cat(self.detection_labels + self.groundtruth_labels).unique().cpu().tolist()
        return []

    def _get_coco_format(
        self,
        boxes: List[torch.Tensor],
        labels: List[torch.Tensor],
        scores: Optional[List[torch.Tensor]] = None,
        crowds: Optional[List[torch.Tensor]] = None,
        area: Optional[List[torch.Tensor]] = None,
    ) -> Dict:
        """Transforms and returns all cached targets or predictions in COCO format.

        Format is defined at https://cocodataset.org/#format-data
        """
        images = []
        annotations = []
        annotation_id = 1  # has to start with 1, otherwise COCOEval results are wrong

        for image_id, (image_boxes, image_labels) in enumerate(zip(boxes, labels)):
            if self.iou_type == "segm" and len(image_boxes) == 0:
                continue

            if self.iou_type == "bbox":
                image_boxes = image_boxes.cpu().tolist()
            image_labels = image_labels.cpu().tolist()

            images.append({"id": image_id})
            if self.iou_type == "segm":
                images[-1]["height"], images[-1]["width"] = image_boxes[0][0][0], image_boxes[0][0][1]

            for k, (image_box, image_label) in enumerate(zip(image_boxes, image_labels)):
                if self.iou_type == "bbox" and len(image_box) != 4:
                    raise ValueError(
                        f"Invalid input box of sample {image_id}, element {k} (expected 4 values, got {len(image_box)})"
                    )

                if type(image_label) != int:
                    raise ValueError(
                        f"Invalid input class of sample {image_id}, element {k}"
                        f" (expected value of type integer, got type {type(image_label)})"
                    )

                stat = image_box if self.iou_type == "bbox" else {"size": image_box[0], "counts": image_box[1]}

                if area is not None and area[image_id][k].cpu().tolist() > 0:
                    area_stat = area[image_id][k].cpu().tolist()
                else:
                    area_stat = image_box[2] * image_box[3] if self.iou_type == "bbox" else mask_utils.area(stat)

                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "bbox" if self.iou_type == "bbox" else "segmentation": stat,
                    "area": area_stat,
                    "category_id": image_label,
                    "iscrowd": crowds[image_id][k].cpu().tolist() if crowds is not None else 0,
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

        classes = [{"id": i, "name": str(i)} for i in self._get_classes()]
        return {"images": images, "annotations": annotations, "categories": classes}

    def plot(
        self, val: Optional[Union[Dict[str, Tensor], Sequence[Dict[str, Tensor]]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure object and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> from torch import tensor
            >>> from torchmetrics.detection.mean_ap import MeanAveragePrecision
            >>> preds = [dict(
            ...     boxes=tensor([[258.0, 41.0, 606.0, 285.0]]),
            ...     scores=tensor([0.536]),
            ...     labels=tensor([0]),
            ... )]
            >>> target = [dict(
            ...     boxes=tensor([[214.0, 41.0, 562.0, 285.0]]),
            ...     labels=tensor([0]),
            ... )]
            >>> metric = MeanAveragePrecision()
            >>> metric.update(preds, target)
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.detection.mean_ap import MeanAveragePrecision
            >>> preds = lambda: [dict(
            ...     boxes=torch.tensor([[258.0, 41.0, 606.0, 285.0]]) + torch.randint(10, (1,4)),
            ...     scores=torch.tensor([0.536]) + 0.1*torch.rand(1),
            ...     labels=torch.tensor([0]),
            ... )]
            >>> target = [dict(
            ...     boxes=torch.tensor([[214.0, 41.0, 562.0, 285.0]]),
            ...     labels=torch.tensor([0]),
            ... )]
            >>> metric = MeanAveragePrecision()
            >>> vals = []
            >>> for _ in range(20):
            ...     vals.append(metric(preds(), target))
            >>> fig_, ax_ = metric.plot(vals)
        """
        return self._plot(val, ax)

    # --------------------
    # specialized syncronization and apply functions for this metric
    # --------------------

    def _apply(self, fn: Callable) -> torch.nn.Module:
        """Custom apply function.

        Excludes the detections and groundtruths from the casting when the iou_type is set to `segm` as the state is
        no longer a tensor but a tuple.
        """
        if self.iou_type == "segm":
            this = super()._apply(fn, exclude_state=("detections", "groundtruths"))
        else:
            this = super()._apply(fn)
        return this

    def _sync_dist(self, dist_sync_fn: Optional[Callable] = None, process_group: Optional[Any] = None) -> None:
        """Custom sync function.

        For the iou_type `segm` the detections and groundtruths are no longer tensors but tuples. Therefore, we need
        to gather the list of tuples and then convert it back to a list of tuples.

        """
        super()._sync_dist(dist_sync_fn=dist_sync_fn, process_group=process_group)

        if self.iou_type == "segm":
            self.detections = self._gather_tuple_list(self.detections, process_group)
            self.groundtruths = self._gather_tuple_list(self.groundtruths, process_group)

    @staticmethod
    def _gather_tuple_list(list_to_gather: List[Tuple], process_group: Optional[Any] = None) -> List[Any]:
        """Gather a list of tuples over multiple devices.

        Args:
            list_to_gather: input list of tuples that should be gathered across devices
            process_group: process group to gather the list of tuples

        Returns:
            list of tuples gathered across devices

        """
        world_size = dist.get_world_size(group=process_group)
        dist.barrier(group=process_group)

        list_gathered = [None for _ in range(world_size)]
        dist.all_gather_object(list_gathered, list_to_gather, group=process_group)

        list_merged = []
        for idx in range(len(list_gathered[0])):
            for rank in range(world_size):
                list_merged.append(list_gathered[rank][idx])

        return list_merged
