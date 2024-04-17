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
import contextlib
import io
import json
from types import ModuleType
from typing import Any, Callable, ClassVar, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from lightning_utilities import apply_to_collection
from torch import Tensor
from torch import distributed as dist
from typing_extensions import Literal

from torchmetrics.detection.helpers import _fix_empty_tensors, _input_validator, _validate_iou_type_arg
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.imports import (
    _FASTER_COCO_EVAL_AVAILABLE,
    _MATPLOTLIB_AVAILABLE,
    _PYCOCOTOOLS_AVAILABLE,
    _TORCHVISION_GREATER_EQUAL_0_8,
)
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["MeanAveragePrecision.plot"]

if not _TORCHVISION_GREATER_EQUAL_0_8 or not (_PYCOCOTOOLS_AVAILABLE or _FASTER_COCO_EVAL_AVAILABLE):
    __doctest_skip__ = [
        "MeanAveragePrecision.plot",
        "MeanAveragePrecision",
        "MeanAveragePrecision.tm_to_coco",
        "MeanAveragePrecision.coco_to_tm",
    ]


def _load_backend_tools(backend: Literal["pycocotools", "faster_coco_eval"]) -> Tuple[object, object, ModuleType]:
    """Load the backend tools for the given backend."""
    if backend == "pycocotools":
        if not _PYCOCOTOOLS_AVAILABLE:
            raise ModuleNotFoundError(
                "Backend `pycocotools` in metric `MeanAveragePrecision`  metric requires that `pycocotools` is"
                " installed. Please install with `pip install pycocotools` or `pip install torchmetrics[detection]`"
            )
        import pycocotools.mask as mask_utils
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        return COCO, COCOeval, mask_utils

    if not _FASTER_COCO_EVAL_AVAILABLE:
        raise ModuleNotFoundError(
            "Backend `faster_coco_eval` in metric `MeanAveragePrecision`  metric requires that `faster-coco-eval` is"
            " installed. Please install with `pip install faster-coco-eval`."
        )
    from faster_coco_eval import COCO
    from faster_coco_eval import COCOeval_faster as COCOeval
    from faster_coco_eval.core import mask as mask_utils

    return COCO, COCOeval, mask_utils


class MeanAveragePrecision(Metric):
    r"""Compute the `Mean-Average-Precision (mAP) and Mean-Average-Recall (mAR)`_ for object detection predictions.

    .. math::
        \text{mAP} = \frac{1}{n} \sum_{i=1}^{n} AP_i

    where :math:`AP_i` is the average precision for class :math:`i` and :math:`n` is the number of classes. The average
    precision is defined as the area under the precision-recall curve. For object detection the recall and precision are
    defined based on the intersection of union (IoU) between the predicted bounding boxes and the ground truth bounding
    boxes e.g. if two boxes have an IoU > t (with t being some threshold) they are considered a match and therefore
    considered a true positive. The precision is then defined as the number of true positives divided by the number of
    all detected boxes and the recall is defined as the number of true positives divided by the number of all ground
    boxes.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~List`): A list consisting of dictionaries each containing the key-values
      (each dictionary corresponds to a single image). Parameters that should be provided per dict

        - ``boxes`` (:class:`~torch.Tensor`): float tensor of shape ``(num_boxes, 4)`` containing ``num_boxes``
          detection boxes of the format specified in the constructor.
          By default, this method expects ``(xmin, ymin, xmax, ymax)`` in absolute image coordinates, but can be changed
          using the ``box_format`` parameter. Only required when `iou_type="bbox"`.
        - ``scores`` (:class:`~torch.Tensor`): float tensor of shape ``(num_boxes)`` containing detection scores for the
          boxes.
        - ``labels`` (:class:`~torch.Tensor`): integer tensor of shape ``(num_boxes)`` containing 0-indexed detection
          classes for the boxes.
        - ``masks`` (:class:`~torch.Tensor`): boolean tensor of shape ``(num_boxes, image_height, image_width)``
          containing boolean masks. Only required when `iou_type="segm"`.

    - ``target`` (:class:`~List`): A list consisting of dictionaries each containing the key-values
      (each dictionary corresponds to a single image). Parameters that should be provided per dict:

        - ``boxes`` (:class:`~torch.Tensor`): float tensor of shape ``(num_boxes, 4)`` containing ``num_boxes`` ground
          truth boxes of the format specified in the constructor. only required when `iou_type="bbox"`.
          By default, this method expects ``(xmin, ymin, xmax, ymax)`` in absolute image coordinates.
        - ``labels`` (:class:`~torch.Tensor`): integer tensor of shape ``(num_boxes)`` containing 0-indexed ground truth
          classes for the boxes.
        - ``masks`` (:class:`~torch.Tensor`): boolean tensor of shape ``(num_boxes, image_height, image_width)``
          containing boolean masks. Only required when `iou_type="segm"`.
        - ``iscrowd`` (:class:`~torch.Tensor`): integer tensor of shape ``(num_boxes)`` containing 0/1 values indicating
          whether the bounding box/masks indicate a crowd of objects. Value is optional, and if not provided it will
          automatically be set to 0.
        - ``area`` (:class:`~torch.Tensor`): float tensor of shape ``(num_boxes)`` containing the area of the object.
          Value is optional, and if not provided will be automatically calculated based on the bounding box/masks
          provided. Only affects which samples contribute to the `map_small`, `map_medium`, `map_large` values

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``map_dict``: A dictionary containing the following key-values:

        - map: (:class:`~torch.Tensor`), global mean average precision
        - map_small: (:class:`~torch.Tensor`), mean average precision for small objects
        - map_medium:(:class:`~torch.Tensor`), mean average precision for medium objects
        - map_large: (:class:`~torch.Tensor`), mean average precision for large objects
        - mar_{mdt[0]}: (:class:`~torch.Tensor`), mean average recall for `max_detection_thresholds[0]` (default 1)
          detection per image
        - mar_{mdt[1]}: (:class:`~torch.Tensor`), mean average recall for `max_detection_thresholds[1]` (default 10)
          detection per image
        - mar_{mdt[1]}: (:class:`~torch.Tensor`), mean average recall for `max_detection_thresholds[2]` (default 100)
          detection per image
        - mar_small: (:class:`~torch.Tensor`), mean average recall for small objects
        - mar_medium: (:class:`~torch.Tensor`), mean average recall for medium objects
        - mar_large: (:class:`~torch.Tensor`), mean average recall for large objects
        - map_50: (:class:`~torch.Tensor`) (-1 if 0.5 not in the list of iou thresholds), mean average precision at
          IoU=0.50
        - map_75: (:class:`~torch.Tensor`) (-1 if 0.75 not in the list of iou thresholds), mean average precision at
          IoU=0.75
        - map_per_class: (:class:`~torch.Tensor`) (-1 if class metrics are disabled), mean average precision per
          observed class
        - mar_{mdt[2]}_per_class: (:class:`~torch.Tensor`) (-1 if class metrics are disabled), mean average recall for
          `max_detection_thresholds[2]` (default 100) detections per image per observed class
        - classes (:class:`~torch.Tensor`), list of all observed classes

    For an example on how to use this metric check the `torchmetrics mAP example`_.

    .. note::
        ``map`` score is calculated with @[ IoU=self.iou_thresholds | area=all | max_dets=max_detection_thresholds ].
        Caution: If the initialization parameters are changed, dictionary keys for mAR can change as well.

    .. note::
        This metric supports, at the moment, two different backends for the evaluation. The default backend is
        ``"pycocotools"``, which either require the official `pycocotools`_ implementation or this
        `fork of pycocotools`_ to be installed. We recommend using the fork as it is better maintained and easily
        available to install via pip: `pip install pycocotools`. It is also this fork that will be installed if you
        install ``torchmetrics[detection]``. The second backend is the `faster-coco-eval`_ implementation, which can be
        installed with ``pip install faster-coco-eval``. This implementation is a maintained open-source implementation
        that is faster and corrects certain corner cases that the official implementation has. Our own testing has shown
        that the results are identical to the official implementation. Regardless of the backend we also require you to
        have `torchvision` version 0.8.0 or newer installed. Please install with ``pip install torchvision>=0.8`` or
        ``pip install torchmetrics[detection]``.

    Args:
        box_format:
            Input format of given boxes. Supported formats are:

                - 'xyxy': boxes are represented via corners, x1, y1 being top left and x2, y2 being bottom right.
                - 'xywh' : boxes are represented via corner, width and height, x1, y2 being top left, w, h being
                  width and height. This is the default format used by pycoco and all input formats will be converted
                  to this.
                - 'cxcywh': boxes are represented via centre, width and height, cx, cy being center of box, w, h being
                  width and height.

        iou_type:
            Type of input (either masks or bounding-boxes) used for computing IOU. Supported IOU types are
            ``"bbox"`` or ``"segm"`` or both as a tuple.
        iou_thresholds:
            IoU thresholds for evaluation. If set to ``None`` it corresponds to the stepped range ``[0.5,...,0.95]``
            with step ``0.05``. Else provide a list of floats.
        rec_thresholds:
            Recall thresholds for evaluation. If set to ``None`` it corresponds to the stepped range ``[0,...,1]``
            with step ``0.01``. Else provide a list of floats.
        max_detection_thresholds:
            Thresholds on max detections per image. If set to `None` will use thresholds ``[1, 10, 100]``.
            Else, please provide a list of ints of length 3, which is the only supported length by both backends.
        class_metrics:
            Option to enable per-class metrics for mAP and mAR_100. Has a performance impact that scales linearly with
            the number of classes in the dataset.
        extended_summary:
            Option to enable extended summary with additional metrics including IOU, precision and recall. The output
            dictionary will contain the following extra key-values:

                - ``ious``: a dictionary containing the IoU values for every image/class combination e.g.
                  ``ious[(0,0)]`` would contain the IoU for image 0 and class 0. Each value is a tensor with shape
                  ``(n,m)`` where ``n`` is the number of detections and ``m`` is the number of ground truth boxes for
                  that image/class combination.
                - ``precision``: a tensor of shape ``(TxRxKxAxM)`` containing the precision values. Here ``T`` is the
                  number of IoU thresholds, ``R`` is the number of recall thresholds, ``K`` is the number of classes,
                  ``A`` is the number of areas and ``M`` is the number of max detections per image.
                - ``recall``: a tensor of shape ``(TxKxAxM)`` containing the recall values. Here ``T`` is the number of
                  IoU thresholds, ``K`` is the number of classes, ``A`` is the number of areas and ``M`` is the number
                  of max detections per image.
                - ``scores``: a tensor of shape ``(TxRxKxAxM)`` containing the confidence scores.  Here ``T`` is the
                  number of IoU thresholds, ``R`` is the number of recall thresholds, ``K`` is the number of classes,
                  ``A`` is the number of areas and ``M`` is the number of max detections per image.

        average:
            Method for averaging scores over labels. Choose between "``"macro"`` and ``"micro"``.
        backend:
            Backend to use for the evaluation. Choose between ``"pycocotools"`` and ``"faster_coco_eval"``.

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ModuleNotFoundError:
            If ``pycocotools`` is not installed
        ModuleNotFoundError:
            If ``torchvision`` is not installed or version installed is lower than 0.8.0
        ValueError:
            If ``box_format`` is not one of ``"xyxy"``, ``"xywh"`` or ``"cxcywh"``
        ValueError:
            If ``iou_type`` is not one of ``"bbox"`` or ``"segm"``
        ValueError:
            If ``iou_thresholds`` is not None or a list of floats
        ValueError:
            If ``rec_thresholds`` is not None or a list of floats
        ValueError:
            If ``max_detection_thresholds`` is not None or a list of ints
        ValueError:
            If ``class_metrics`` is not a boolean

    Example::

        Basic example for when `iou_type="bbox"`. In this case the ``boxes`` key is required in the input dictionaries,
        in addition to the ``scores`` and ``labels`` keys.

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
        >>> metric = MeanAveragePrecision(iou_type="bbox")
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

    Example::

        Basic example for when `iou_type="segm"`. In this case the ``masks`` key is required in the input dictionaries,
        in addition to the ``scores`` and ``labels`` keys.

        >>> from torch import tensor
        >>> from torchmetrics.detection import MeanAveragePrecision
        >>> mask_pred = [
        ...   [0, 0, 0, 0, 0],
        ...   [0, 0, 1, 1, 0],
        ...   [0, 0, 1, 1, 0],
        ...   [0, 0, 0, 0, 0],
        ...   [0, 0, 0, 0, 0],
        ... ]
        >>> mask_tgt = [
        ...   [0, 0, 0, 0, 0],
        ...   [0, 0, 1, 0, 0],
        ...   [0, 0, 1, 1, 0],
        ...   [0, 0, 1, 0, 0],
        ...   [0, 0, 0, 0, 0],
        ... ]
        >>> preds = [
        ...   dict(
        ...     masks=tensor([mask_pred], dtype=torch.bool),
        ...     scores=tensor([0.536]),
        ...     labels=tensor([0]),
        ...   )
        ... ]
        >>> target = [
        ...   dict(
        ...     masks=tensor([mask_tgt], dtype=torch.bool),
        ...     labels=tensor([0]),
        ...   )
        ... ]
        >>> metric = MeanAveragePrecision(iou_type="segm")
        >>> metric.update(preds, target)
        >>> from pprint import pprint
        >>> pprint(metric.compute())
        {'classes': tensor(0, dtype=torch.int32),
         'map': tensor(0.2000),
         'map_50': tensor(1.),
         'map_75': tensor(0.),
         'map_large': tensor(-1.),
         'map_medium': tensor(-1.),
         'map_per_class': tensor(-1.),
         'map_small': tensor(0.2000),
         'mar_1': tensor(0.2000),
         'mar_10': tensor(0.2000),
         'mar_100': tensor(0.2000),
         'mar_100_per_class': tensor(-1.),
         'mar_large': tensor(-1.),
         'mar_medium': tensor(-1.),
         'mar_small': tensor(0.2000)}

    """

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = True
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    detection_box: List[Tensor]
    detection_mask: List[Tensor]
    detection_scores: List[Tensor]
    detection_labels: List[Tensor]
    groundtruth_box: List[Tensor]
    groundtruth_mask: List[Tensor]
    groundtruth_labels: List[Tensor]
    groundtruth_crowds: List[Tensor]
    groundtruth_area: List[Tensor]

    warn_on_many_detections: bool = True

    __jit_unused_properties__: ClassVar[List[str]] = [
        "is_differentiable",
        "higher_is_better",
        "plot_lower_bound",
        "plot_upper_bound",
        "plot_legend_name",
        "metric_state",
        "_update_called",
        # below is added for specifically for this metric
        "coco",
        "cocoeval",
        "mask_utils",
    ]

    def __init__(
        self,
        box_format: Literal["xyxy", "xywh", "cxcywh"] = "xyxy",
        iou_type: Union[Literal["bbox", "segm"], Tuple[str]] = "bbox",
        iou_thresholds: Optional[List[float]] = None,
        rec_thresholds: Optional[List[float]] = None,
        max_detection_thresholds: Optional[List[int]] = None,
        class_metrics: bool = False,
        extended_summary: bool = False,
        average: Literal["macro", "micro"] = "macro",
        backend: Literal["pycocotools", "faster_coco_eval"] = "pycocotools",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not (_PYCOCOTOOLS_AVAILABLE or _FASTER_COCO_EVAL_AVAILABLE):
            raise ModuleNotFoundError(
                "`MAP` metric requires that `pycocotools` or `faster-coco-eval` installed."
                " Please install with `pip install pycocotools` or `pip install faster-coco-eval` or"
                " `pip install torchmetrics[detection]`."
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

        self.iou_type = _validate_iou_type_arg(iou_type)

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
        if max_detection_thresholds is not None and len(max_detection_thresholds) != 3:
            raise ValueError(
                "When providing a list of max detection thresholds it should have length 3."
                " Got value {len(max_detection_thresholds)}"
            )
        max_det_threshold, _ = torch.sort(torch.tensor(max_detection_thresholds or [1, 10, 100], dtype=torch.int))
        self.max_detection_thresholds = max_det_threshold.tolist()

        if not isinstance(class_metrics, bool):
            raise ValueError("Expected argument `class_metrics` to be a boolean")
        self.class_metrics = class_metrics

        if not isinstance(extended_summary, bool):
            raise ValueError("Expected argument `extended_summary` to be a boolean")
        self.extended_summary = extended_summary

        if average not in ("macro", "micro"):
            raise ValueError(f"Expected argument `average` to be one of ('macro', 'micro') but got {average}")
        self.average = average

        if backend not in ("pycocotools", "faster_coco_eval"):
            raise ValueError(
                f"Expected argument `backend` to be one of ('pycocotools', 'faster_coco_eval') but got {backend}"
            )
        self.backend = backend

        self.add_state("detection_box", default=[], dist_reduce_fx=None)
        self.add_state("detection_mask", default=[], dist_reduce_fx=None)
        self.add_state("detection_scores", default=[], dist_reduce_fx=None)
        self.add_state("detection_labels", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_box", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_mask", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_labels", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_crowds", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_area", default=[], dist_reduce_fx=None)

    @property
    def coco(self) -> object:
        """Returns the coco module for the given backend, done in this way to make metric picklable."""
        coco, _, _ = _load_backend_tools(self.backend)
        return coco

    @property
    def cocoeval(self) -> object:
        """Returns the coco eval module for the given backend, done in this way to make metric picklable."""
        _, cocoeval, _ = _load_backend_tools(self.backend)
        return cocoeval

    @property
    def mask_utils(self) -> object:
        """Returns the mask utils object for the given backend, done in this way to make metric picklable."""
        _, _, mask_utils = _load_backend_tools(self.backend)
        return mask_utils

    def update(self, preds: List[Dict[str, Tensor]], target: List[Dict[str, Tensor]]) -> None:
        """Update metric state.

        Raises:
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

        """
        _input_validator(preds, target, iou_type=self.iou_type)  # type: ignore[arg-type]

        for item in preds:
            bbox_detection, mask_detection = self._get_safe_item_values(item, warn=self.warn_on_many_detections)
            if bbox_detection is not None:
                self.detection_box.append(bbox_detection)
            if mask_detection is not None:
                self.detection_mask.append(mask_detection)  # type: ignore[arg-type]
            self.detection_labels.append(item["labels"])
            self.detection_scores.append(item["scores"])

        for item in target:
            bbox_groundtruth, mask_groundtruth = self._get_safe_item_values(item)
            if bbox_groundtruth is not None:
                self.groundtruth_box.append(bbox_groundtruth)
            if mask_groundtruth is not None:
                self.groundtruth_mask.append(mask_groundtruth)  # type: ignore[arg-type]
            self.groundtruth_labels.append(item["labels"])
            self.groundtruth_crowds.append(item.get("iscrowd", torch.zeros_like(item["labels"])))
            self.groundtruth_area.append(item.get("area", torch.zeros_like(item["labels"])))

    def compute(self) -> dict:
        """Computes the metric."""
        coco_preds, coco_target = self._get_coco_datasets(average=self.average)

        result_dict = {}
        with contextlib.redirect_stdout(io.StringIO()):
            for i_type in self.iou_type:
                prefix = "" if len(self.iou_type) == 1 else f"{i_type}_"
                if len(self.iou_type) > 1:
                    # the area calculation is different for bbox and segm and therefore to get the small, medium and
                    # large values correct we need to dynamically change the area attribute of the annotations
                    for anno in coco_preds.dataset["annotations"]:
                        anno["area"] = anno[f"area_{i_type}"]

                coco_eval = self.cocoeval(coco_target, coco_preds, iouType=i_type)  # type: ignore[operator]
                coco_eval.params.iouThrs = np.array(self.iou_thresholds, dtype=np.float64)
                coco_eval.params.recThrs = np.array(self.rec_thresholds, dtype=np.float64)
                coco_eval.params.maxDets = self.max_detection_thresholds

                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                stats = coco_eval.stats
                result_dict.update(self._coco_stats_to_tensor_dict(stats, prefix=prefix))

                summary = {}
                if self.extended_summary:
                    summary = {
                        f"{prefix}ious": apply_to_collection(
                            coco_eval.ious, np.ndarray, lambda x: torch.tensor(x, dtype=torch.float32)
                        ),
                        f"{prefix}precision": torch.tensor(coco_eval.eval["precision"]),
                        f"{prefix}recall": torch.tensor(coco_eval.eval["recall"]),
                        f"{prefix}scores": torch.tensor(coco_eval.eval["scores"]),
                    }
                result_dict.update(summary)

                # if class mode is enabled, evaluate metrics per class
                if self.class_metrics:
                    if self.average == "micro":
                        # since micro averaging have all the data in one class, we need to reinitialize the coco_eval
                        # object in macro mode to get the per class stats
                        coco_preds, coco_target = self._get_coco_datasets(average="macro")
                        coco_eval = self.cocoeval(coco_target, coco_preds, iouType=i_type)  # type: ignore[operator]
                        coco_eval.params.iouThrs = np.array(self.iou_thresholds, dtype=np.float64)
                        coco_eval.params.recThrs = np.array(self.rec_thresholds, dtype=np.float64)
                        coco_eval.params.maxDets = self.max_detection_thresholds

                    map_per_class_list = []
                    mar_per_class_list = []
                    for class_id in self._get_classes():
                        coco_eval.params.catIds = [class_id]
                        with contextlib.redirect_stdout(io.StringIO()):
                            coco_eval.evaluate()
                            coco_eval.accumulate()
                            coco_eval.summarize()
                            class_stats = coco_eval.stats

                        map_per_class_list.append(torch.tensor([class_stats[0]]))
                        mar_per_class_list.append(torch.tensor([class_stats[8]]))

                    map_per_class_values = torch.tensor(map_per_class_list, dtype=torch.float32)
                    mar_per_class_values = torch.tensor(mar_per_class_list, dtype=torch.float32)
                else:
                    map_per_class_values = torch.tensor([-1], dtype=torch.float32)
                    mar_per_class_values = torch.tensor([-1], dtype=torch.float32)
                prefix = "" if len(self.iou_type) == 1 else f"{i_type}_"
                result_dict.update(
                    {
                        f"{prefix}map_per_class": map_per_class_values,
                        f"{prefix}mar_{self.max_detection_thresholds[-1]}_per_class": mar_per_class_values,
                    },
                )
        result_dict.update({"classes": torch.tensor(self._get_classes(), dtype=torch.int32)})

        return result_dict

    def _get_coco_datasets(self, average: Literal["macro", "micro"]) -> Tuple[object, object]:
        """Returns the coco datasets for the target and the predictions."""
        if average == "micro":
            # for micro averaging we set everything to be the same class
            groundtruth_labels = apply_to_collection(self.groundtruth_labels, Tensor, lambda x: torch.zeros_like(x))
            detection_labels = apply_to_collection(self.detection_labels, Tensor, lambda x: torch.zeros_like(x))
        else:
            groundtruth_labels = self.groundtruth_labels
            detection_labels = self.detection_labels

        coco_target, coco_preds = self.coco(), self.coco()  # type: ignore[operator]

        coco_target.dataset = self._get_coco_format(
            labels=groundtruth_labels,
            boxes=self.groundtruth_box if len(self.groundtruth_box) > 0 else None,
            masks=self.groundtruth_mask if len(self.groundtruth_mask) > 0 else None,
            crowds=self.groundtruth_crowds,
            area=self.groundtruth_area,
        )
        coco_preds.dataset = self._get_coco_format(
            labels=detection_labels,
            boxes=self.detection_box if len(self.detection_box) > 0 else None,
            masks=self.detection_mask if len(self.detection_mask) > 0 else None,
            scores=self.detection_scores,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            coco_target.createIndex()
            coco_preds.createIndex()

        return coco_preds, coco_target

    def _coco_stats_to_tensor_dict(self, stats: List[float], prefix: str) -> Dict[str, Tensor]:
        """Converts the output of COCOeval.stats to a dict of tensors."""
        mdt = self.max_detection_thresholds
        return {
            f"{prefix}map": torch.tensor([stats[0]], dtype=torch.float32),
            f"{prefix}map_50": torch.tensor([stats[1]], dtype=torch.float32),
            f"{prefix}map_75": torch.tensor([stats[2]], dtype=torch.float32),
            f"{prefix}map_small": torch.tensor([stats[3]], dtype=torch.float32),
            f"{prefix}map_medium": torch.tensor([stats[4]], dtype=torch.float32),
            f"{prefix}map_large": torch.tensor([stats[5]], dtype=torch.float32),
            f"{prefix}mar_{mdt[0]}": torch.tensor([stats[6]], dtype=torch.float32),
            f"{prefix}mar_{mdt[1]}": torch.tensor([stats[7]], dtype=torch.float32),
            f"{prefix}mar_{mdt[2]}": torch.tensor([stats[8]], dtype=torch.float32),
            f"{prefix}mar_small": torch.tensor([stats[9]], dtype=torch.float32),
            f"{prefix}mar_medium": torch.tensor([stats[10]], dtype=torch.float32),
            f"{prefix}mar_large": torch.tensor([stats[11]], dtype=torch.float32),
        }

    @staticmethod
    def coco_to_tm(
        coco_preds: str,
        coco_target: str,
        iou_type: Union[Literal["bbox", "segm"], List[str]] = "bbox",
        backend: Literal["pycocotools", "faster_coco_eval"] = "pycocotools",
    ) -> Tuple[List[Dict[str, Tensor]], List[Dict[str, Tensor]]]:
        """Utility function for converting .json coco format files to the input format of this metric.

        The function accepts a file for the predictions and a file for the target in coco format and converts them to
        a list of dictionaries containing the boxes, labels and scores in the input format of this metric.

        Args:
            coco_preds: Path to the json file containing the predictions in coco format
            coco_target: Path to the json file containing the targets in coco format
            iou_type: Type of input, either `bbox` for bounding boxes or `segm` for segmentation masks
            backend: Backend to use for the conversion. Either `pycocotools` or `faster_coco_eval`.

        Returns:
            A tuple containing the predictions and targets in the input format of this metric. Each element of the
            tuple is a list of dictionaries containing the boxes, labels and scores.

        Example:
            >>> # File formats are defined at https://cocodataset.org/#format-data
            >>> # Example files can be found at
            >>> # https://github.com/cocodataset/cocoapi/tree/master/results
            >>> from torchmetrics.detection import MeanAveragePrecision
            >>> preds, target = MeanAveragePrecision.coco_to_tm(
            ...   "instances_val2014_fakebbox100_results.json.json",
            ...   "val2014_fake_eval_res.txt.json"
            ...   iou_type="bbox"
            ... )  # doctest: +SKIP

        """
        iou_type = _validate_iou_type_arg(iou_type)  # type: ignore[arg-type]
        coco, _, _ = _load_backend_tools(backend)

        with contextlib.redirect_stdout(io.StringIO()):
            gt = coco(coco_target)  # type: ignore[operator]
            dt = gt.loadRes(coco_preds)

        gt_dataset = gt.dataset["annotations"]
        dt_dataset = dt.dataset["annotations"]

        target: dict = {}
        for t in gt_dataset:
            if t["image_id"] not in target:
                target[t["image_id"]] = {
                    "labels": [],
                    "iscrowd": [],
                    "area": [],
                }
                if "bbox" in iou_type:
                    target[t["image_id"]]["boxes"] = []
                if "segm" in iou_type:
                    target[t["image_id"]]["masks"] = []

            if "bbox" in iou_type:
                target[t["image_id"]]["boxes"].append(t["bbox"])
            if "segm" in iou_type:
                target[t["image_id"]]["masks"].append(gt.annToMask(t))
            target[t["image_id"]]["labels"].append(t["category_id"])
            target[t["image_id"]]["iscrowd"].append(t["iscrowd"])
            target[t["image_id"]]["area"].append(t["area"])

        preds: dict = {}
        for p in dt_dataset:
            if p["image_id"] not in preds:
                preds[p["image_id"]] = {"scores": [], "labels": []}
                if "bbox" in iou_type:
                    preds[p["image_id"]]["boxes"] = []
                if "segm" in iou_type:
                    preds[p["image_id"]]["masks"] = []
            if "bbox" in iou_type:
                preds[p["image_id"]]["boxes"].append(p["bbox"])
            if "segm" in iou_type:
                preds[p["image_id"]]["masks"].append(gt.annToMask(p))
            preds[p["image_id"]]["scores"].append(p["score"])
            preds[p["image_id"]]["labels"].append(p["category_id"])
        for k in target:  # add empty predictions for images without predictions
            if k not in preds:
                preds[k] = {"scores": [], "labels": []}
                if "bbox" in iou_type:
                    preds[k]["boxes"] = []
                if "segm" in iou_type:
                    preds[k]["masks"] = []

        batched_preds, batched_target = [], []
        for key in target:
            bp = {
                "scores": torch.tensor(preds[key]["scores"], dtype=torch.float32),
                "labels": torch.tensor(preds[key]["labels"], dtype=torch.int32),
            }
            if "bbox" in iou_type:
                bp["boxes"] = torch.tensor(np.array(preds[key]["boxes"]), dtype=torch.float32)
            if "segm" in iou_type:
                bp["masks"] = torch.tensor(np.array(preds[key]["masks"]), dtype=torch.uint8)
            batched_preds.append(bp)

            bt = {
                "labels": torch.tensor(target[key]["labels"], dtype=torch.int32),
                "iscrowd": torch.tensor(target[key]["iscrowd"], dtype=torch.int32),
                "area": torch.tensor(target[key]["area"], dtype=torch.float32),
            }
            if "bbox" in iou_type:
                bt["boxes"] = torch.tensor(target[key]["boxes"], dtype=torch.float32)
            if "segm" in iou_type:
                bt["masks"] = torch.tensor(np.array(target[key]["masks"]), dtype=torch.uint8)
            batched_target.append(bt)

        return batched_preds, batched_target

    def tm_to_coco(self, name: str = "tm_map_input") -> None:
        """Utility function for converting the input for this metric to coco format and saving it to a json file.

        This function should be used after calling `.update(...)` or `.forward(...)` on all data that should be written
        to the file, as the input is then internally cached. The function then converts to information to coco format
        a writes it to json files.

        Args:
            name: Name of the output file, which will be appended with "_preds.json" and "_target.json"

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
            >>> metric.tm_to_coco("tm_map_input")  # doctest: +SKIP

        """
        target_dataset = self._get_coco_format(
            labels=self.groundtruth_labels,
            boxes=self.groundtruth_box,
            masks=self.groundtruth_mask,
            crowds=self.groundtruth_crowds,
            area=self.groundtruth_area,
        )
        preds_dataset = self._get_coco_format(
            labels=self.detection_labels, boxes=self.detection_box, masks=self.detection_mask
        )

        preds_json = json.dumps(preds_dataset["annotations"], indent=4)
        target_json = json.dumps(target_dataset, indent=4)

        with open(f"{name}_preds.json", "w") as f:
            f.write(preds_json)

        with open(f"{name}_target.json", "w") as f:
            f.write(target_json)

    def _get_safe_item_values(
        self, item: Dict[str, Any], warn: bool = False
    ) -> Tuple[Optional[Tensor], Optional[Tuple]]:
        """Convert and return the boxes or masks from the item depending on the iou_type.

        Args:
            item: input dictionary containing the boxes or masks
            warn: whether to warn if the number of boxes or masks exceeds the max_detection_thresholds

        Returns:
            boxes or masks depending on the iou_type

        """
        from torchvision.ops import box_convert

        output = [None, None]
        if "bbox" in self.iou_type:
            boxes = _fix_empty_tensors(item["boxes"])
            if boxes.numel() > 0:
                boxes = box_convert(boxes, in_fmt=self.box_format, out_fmt="xywh")
            output[0] = boxes  # type: ignore[call-overload]
        if "segm" in self.iou_type:
            masks = []
            for i in item["masks"].cpu().numpy():
                rle = self.mask_utils.encode(np.asfortranarray(i))
                masks.append((tuple(rle["size"]), rle["counts"]))
            output[1] = tuple(masks)  # type: ignore[call-overload]
        if warn and (
            (output[0] is not None and len(output[0]) > self.max_detection_thresholds[-1])
            or (output[1] is not None and len(output[1]) > self.max_detection_thresholds[-1])
        ):
            _warning_on_too_many_detections(self.max_detection_thresholds[-1])
        return output  # type: ignore[return-value]

    def _get_classes(self) -> List:
        """Return a list of unique classes found in ground truth and detection data."""
        if len(self.detection_labels) > 0 or len(self.groundtruth_labels) > 0:
            return torch.cat(self.detection_labels + self.groundtruth_labels).unique().cpu().tolist()
        return []

    def _get_coco_format(
        self,
        labels: List[torch.Tensor],
        boxes: Optional[List[torch.Tensor]] = None,
        masks: Optional[List[torch.Tensor]] = None,
        scores: Optional[List[torch.Tensor]] = None,
        crowds: Optional[List[torch.Tensor]] = None,
        area: Optional[List[torch.Tensor]] = None,
    ) -> Dict:
        """Transforms and returns all cached targets or predictions in COCO format.

        Format is defined at
        https://cocodataset.org/#format-data

        """
        images = []
        annotations = []
        annotation_id = 1  # has to start with 1, otherwise COCOEval results are wrong

        for image_id, image_labels in enumerate(labels):
            if boxes is not None:
                image_boxes = boxes[image_id]
                image_boxes = image_boxes.cpu().tolist()
            if masks is not None:
                image_masks = masks[image_id]
                if len(image_masks) == 0 and boxes is None:
                    continue
            image_labels = image_labels.cpu().tolist()  # type: ignore[assignment]

            images.append({"id": image_id})
            if "segm" in self.iou_type and len(image_masks) > 0:
                images[-1]["height"], images[-1]["width"] = image_masks[0][0][0], image_masks[0][0][1]  # type: ignore[assignment]

            for k, image_label in enumerate(image_labels):
                if boxes is not None:
                    image_box = image_boxes[k]
                if masks is not None and len(image_masks) > 0:
                    image_mask = image_masks[k]
                    image_mask = {"size": image_mask[0], "counts": image_mask[1]}

                if "bbox" in self.iou_type and len(image_box) != 4:
                    raise ValueError(
                        f"Invalid input box of sample {image_id}, element {k} (expected 4 values, got {len(image_box)})"
                    )

                if not isinstance(image_label, int):
                    raise ValueError(
                        f"Invalid input class of sample {image_id}, element {k}"
                        f" (expected value of type integer, got type {type(image_label)})"
                    )

                area_stat_box = None
                area_stat_mask = None
                if area is not None and area[image_id][k].cpu().tolist() > 0:  # type: ignore[operator]
                    area_stat = area[image_id][k].cpu().tolist()
                else:
                    area_stat = (
                        self.mask_utils.area(image_mask) if "segm" in self.iou_type else image_box[2] * image_box[3]
                    )
                    if len(self.iou_type) > 1:
                        area_stat_box = image_box[2] * image_box[3]
                        area_stat_mask = self.mask_utils.area(image_mask)

                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "area": area_stat,
                    "category_id": image_label,
                    "iscrowd": crowds[image_id][k].cpu().tolist() if crowds is not None else 0,
                }
                if area_stat_box is not None:
                    annotation["area_bbox"] = area_stat_box
                    annotation["area_segm"] = area_stat_mask

                if boxes is not None:
                    annotation["bbox"] = image_box
                if masks is not None:
                    annotation["segmentation"] = image_mask

                if scores is not None:
                    score = scores[image_id][k].cpu().tolist()
                    if not isinstance(score, float):
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
    # specialized synchronization and apply functions for this metric
    # --------------------

    def _apply(self, fn: Callable) -> torch.nn.Module:  # type: ignore[override]
        """Custom apply function.

        Excludes the detections and groundtruths from the casting when the iou_type is set to `segm` as the state is
        no longer a tensor but a tuple.

        """
        return super()._apply(fn, exclude_state=("detection_mask", "groundtruth_mask"))

    def _sync_dist(self, dist_sync_fn: Optional[Callable] = None, process_group: Optional[Any] = None) -> None:
        """Custom sync function.

        For the iou_type `segm` the detections and groundtruths are no longer tensors but tuples. Therefore, we need
        to gather the list of tuples and then convert it back to a list of tuples.

        """
        super()._sync_dist(dist_sync_fn=dist_sync_fn, process_group=process_group)  # type: ignore[arg-type]

        if "segm" in self.iou_type:
            self.detection_mask = self._gather_tuple_list(self.detection_mask, process_group)  # type: ignore[arg-type]
            self.groundtruth_mask = self._gather_tuple_list(self.groundtruth_mask, process_group)  # type: ignore[arg-type]

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

        return [list_gathered[rank][idx] for idx in range(len(list_gathered[0])) for rank in range(world_size)]  # type: ignore[arg-type,index]


def _warning_on_too_many_detections(limit: int) -> None:
    rank_zero_warn(
        f"Encountered more than {limit} detections in a single image. This means that certain detections with the"
        " lowest scores will be ignored, that may have an undesirable impact on performance. Please consider adjusting"
        " the `max_detection_threshold` to suit your use case. To disable this warning, set attribute class"
        " `warn_on_many_detections=False`, after initializing the metric.",
        UserWarning,
    )
