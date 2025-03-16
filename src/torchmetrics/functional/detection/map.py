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
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from lightning_utilities import apply_to_collection
from torch import Tensor

from torchmetrics.detection.helpers import CocoBackend, _get_safe_item_values, _input_validator, _validate_iou_type_arg


def mean_average_precision(
    preds: List[Dict[str, Any]],
    target: List[Dict[str, Any]],
    box_format: Literal["xyxy", "xywh", "cxcywh"] = "xyxy",
    iou_type: Union[Literal["bbox", "segm"], Tuple[Literal["bbox", "segm"], ...]] = "bbox",
    iou_thresholds: Optional[list[float]] = None,
    rec_thresholds: Optional[list[float]] = None,
    max_detection_thresholds: Optional[list[int]] = None,
    class_metrics: bool = False,
    extended_summary: bool = False,
    average: Literal["macro", "micro"] = "macro",
    backend: Literal["pycocotools", "faster_coco_eval"] = "pycocotools",
    warn_on_many_detections: bool = True,
) -> Union[Tensor, Dict[str, Tensor]]:
    r"""Compute the mean average precision (mAP) and mean average recall (mAR) for object detection predictions.

    This function evaluates detection predictions for either bounding boxes or segmentation masks based
    on the provided ``iou_type``, comparing predictions (``preds``) and ground truth annotations (``target``)
    using a COCO-style evaluation. The expected input for each image is a dictionary with keys:

    - For bounding boxes (``iou_type="bbox"``): ``boxes``, ``scores``, and ``labels``.
    - For segmentation (``iou_type="segm"``): ``masks``, ``scores``, and ``labels``.

    In addition, ground truth dictionaries may include the optional keys ``iscrowd`` and ``area``.
    Boxes are expected in the coordinate format provided via ``box_format``, which supports:
    - ``"xyxy"``: [xmin, ymin, xmax, ymax]
    - ``"xywh"``: [xmin, ymin, width, height]
    - ``"cxcywh"``: [center_x, center_y, width, height]

    The evaluation defaults to IoU thresholds from 0.50 to 0.95 (step 0.05), recall thresholds
    from 0.00 to 1.00 (step 0.01), and maximum detection thresholds of [1, 10, 100]. These can be overridden
    by specifying ``iou_thresholds``, ``rec_thresholds``, and ``max_detection_thresholds``, respectively.
    Optionally, per-class metrics may be computed by enabling ``class_metrics``, and an extended summary
    (including IoU, precision, recall, and scores) is available via ``extended_summary``.
    The averaging method over labels can be set with ``average`` ("macro" or "micro") and the evaluation
    is performed using either the ``pycocotools`` or ``faster_coco_eval`` backend.

    Args:
        preds: List of dictionaries, each representing detection predictions for a single image.
        target: List of dictionaries, each representing ground truth annotations for a single image.
        box_format: Format of the input bounding boxes. Supported values are "xyxy", "xywh", and "cxcywh".
        iou_type: Type of IoU to compute. Can be "bbox", "segm", or a tuple containing both.
        iou_thresholds: List of IoU thresholds (default is [0.5, 0.55, ..., 0.95]).
        rec_thresholds: List of recall thresholds (default is [0.0, 0.01, ..., 1.0]).
        max_detection_thresholds: List of maximum detections per image (default is [1, 10, 100]).
        class_metrics: Whether to compute per-class mAP and mAR metrics.
        extended_summary: Whether to include additional outputs (IoU, precision, recall, scores) in the result.
        average: Averaging method over labels, either "macro" or "micro".
        backend: Backend to use for evaluation ("pycocotools" or "faster_coco_eval").
        warn_on_many_detections: If True, warn when there are an unusually large number of detections.

    Returns:
        dict: A dictionary containing the evaluation metrics. The dictionary includes the following keys:
            - ``map``: Global mean average precision over the defined IoU thresholds.
            - ``mar_{max_det}``: Global mean average recall for each maximum detection threshold.
            - ``map_per_class``: Mean average precision per observed class (or -1 if ``class_metrics`` is disabled).
            - ``mar_{max_det}_per_class``: Mean average recall per observed class for the highest detection threshold.
            - ``classes``: A tensor listing all observed classes.

    Example::

        # Example with bounding boxes
        >>> from torch import tensor
        >>> from torchmetrics.functional.detection.map import mean_average_precision
        >>> preds = [
        ...   {
        ...     "boxes": tensor([[258.0, 41.0, 606.0, 285.0]]),
        ...     "scores": tensor([0.536]),
        ...     "labels": tensor([0]),
        ...   }
        ... ]
        >>> target = [
        ...   {
        ...     "boxes": tensor([[214.0, 41.0, 562.0, 285.0]]),
        ...     "labels": tensor([0]),
        ...   }
        ... ]
        >>> result = mean_average_precision(preds, target, iou_type="bbox")
        >>> print(f"mAP: {result['map']:.4f}, mAP@0.5: {result['map_50']:.4f}")
        mAP: 0.6000, mAP@0.5: 1.0000

    Example::

        # Example with segmentation masks
        >>> import torch
        >>> from torch import tensor
        >>> from torchmetrics.functional.detection.map import mean_average_precision
        >>> mask_pred = tensor([
        ...   [0, 0, 0, 0, 0],
        ...   [0, 0, 1, 1, 0],
        ...   [0, 0, 1, 1, 0],
        ...   [0, 0, 0, 0, 0],
        ...   [0, 0, 0, 0, 0],
        ... ], dtype=torch.bool)
        >>> mask_tgt = tensor([
        ...   [0, 0, 0, 0, 0],
        ...   [0, 0, 1, 0, 0],
        ...   [0, 0, 1, 1, 0],
        ...   [0, 0, 1, 0, 0],
        ...   [0, 0, 0, 0, 0],
        ... ], dtype=torch.bool)
        >>> preds = [
        ...   {
        ...     "masks": mask_pred.unsqueeze(0),
        ...     "scores": tensor([0.536]),
        ...     "labels": tensor([0]),
        ...   }
        ... ]
        >>> target = [
        ...   {
        ...     "masks": mask_tgt.unsqueeze(0),
        ...     "labels": tensor([0]),
        ...   }
        ... ]
        >>> result = mean_average_precision(preds, target, iou_type="segm")
        >>> print(f"mAP: {result['map']:.4f}, mAP@0.5: {result['map_50']:.4f}")
        mAP: 0.2000, mAP@0.5: 1.0000

    """
    iou_thresholds = iou_thresholds or torch.linspace(0.5, 0.95, round((0.95 - 0.5) / 0.05) + 1).tolist()
    max_detection_thresholds = torch.sort(torch.tensor(max_detection_thresholds or [1, 10, 100], dtype=torch.int))[
        0
    ].tolist()
    rec_thresholds = rec_thresholds or torch.linspace(0.0, 1.00, round(1.00 / 0.01) + 1).tolist()

    iou_type = _validate_iou_type_arg(iou_type)
    _input_validator(preds, target, iou_type=iou_type)

    coco_backend = CocoBackend(backend=backend)
    detection_box: List[Tensor] = []
    detection_labels: List[Tensor] = []
    detection_scores: List[Tensor] = []
    detection_mask: List[Tensor] = []
    for item in preds:
        bbox_detection, mask_detection = _get_safe_item_values(
            iou_type, box_format, max_detection_thresholds, coco_backend, item, warn=warn_on_many_detections
        )
        if bbox_detection is not None:
            detection_box.append(bbox_detection)
        if mask_detection is not None:
            detection_mask.append(mask_detection)  # type: ignore[arg-type]
        detection_labels.append(item["labels"])
        detection_scores.append(item["scores"])

    groundtruth_box: List[Tensor] = []
    groundtruth_mask: List[Tensor] = []
    groundtruth_labels: List[Tensor] = []
    groundtruth_crowds: List[Tensor] = []
    groundtruth_area: List[Tensor] = []
    for item in target:
        bbox_groundtruth, mask_groundtruth = _get_safe_item_values(
            iou_type, box_format, max_detection_thresholds, coco_backend, item
        )
        if bbox_groundtruth is not None:
            groundtruth_box.append(bbox_groundtruth)
        if mask_groundtruth is not None:
            groundtruth_mask.append(mask_groundtruth)  # type: ignore[arg-type]
        groundtruth_labels.append(item["labels"])
        groundtruth_crowds.append(item.get("iscrowd", torch.zeros_like(item["labels"])))
        groundtruth_area.append(item.get("area", torch.zeros_like(item["labels"])))

    coco_preds, coco_target = coco_backend._get_coco_datasets(
        groundtruth_labels,
        groundtruth_box,
        groundtruth_mask,
        groundtruth_crowds,
        groundtruth_area,
        detection_labels,
        detection_box,
        detection_mask,
        detection_scores,
        iou_type,
        average=average,
    )

    def get_classes(detection_labels: List[Tensor], groundtruth_labels: List[Tensor]) -> List[int]:
        if len(detection_labels) > 0 or len(groundtruth_labels) > 0:
            return torch.cat(detection_labels + groundtruth_labels).unique().cpu().tolist()
        return []

    result_dict = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for i_type in iou_type:
            prefix = "" if len(iou_type) == 1 else f"{i_type}_"
            if len(iou_type) > 1:
                # the area calculation is different for bbox and segm and therefore to get the small, medium and
                # large values correct we need to dynamically change the area attribute of the annotations
                for anno in coco_preds.dataset["annotations"]:
                    anno["area"] = anno[f"area_{i_type}"]

            if len(coco_preds.imgs) == 0 or len(coco_target.imgs) == 0:
                result_dict.update(
                    coco_backend._coco_stats_to_tensor_dict(
                        12 * [-1.0], prefix=prefix, max_detection_thresholds=max_detection_thresholds
                    )
                )
            else:
                coco_eval = coco_backend.cocoeval(coco_target, coco_preds, iouType=i_type)  # type: ignore[operator]
                coco_eval.params.iouThrs = np.array(iou_thresholds, dtype=np.float64)
                coco_eval.params.recThrs = np.array(rec_thresholds, dtype=np.float64)
                coco_eval.params.maxDets = max_detection_thresholds

                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                stats = coco_eval.stats
                result_dict.update(
                    coco_backend._coco_stats_to_tensor_dict(
                        stats, prefix=prefix, max_detection_thresholds=max_detection_thresholds
                    )
                )

                summary = {}
                if extended_summary:
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
                if class_metrics:
                    # regardless of average method, reinitialize dataset to get rid of internal state which can
                    # lead to wrong results when evaluating per class
                    coco_preds, coco_target = coco_backend._get_coco_datasets(
                        groundtruth_labels,
                        groundtruth_box,
                        groundtruth_mask,
                        groundtruth_crowds,
                        groundtruth_area,
                        detection_labels,
                        detection_box,
                        detection_mask,
                        detection_scores,
                        iou_type,
                        average="macro",
                    )
                    coco_eval = coco_backend.cocoeval(coco_target, coco_preds, iouType=i_type)  # type: ignore[operator]
                    coco_eval.params.iouThrs = np.array(iou_thresholds, dtype=np.float64)
                    coco_eval.params.recThrs = np.array(rec_thresholds, dtype=np.float64)
                    coco_eval.params.maxDets = max_detection_thresholds

                    map_per_class_list = []
                    mar_per_class_list = []
                    for class_id in get_classes(
                        detection_labels=detection_labels, groundtruth_labels=groundtruth_labels
                    ):
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
                prefix = "" if len(iou_type) == 1 else f"{i_type}_"
                result_dict.update(
                    {
                        f"{prefix}map_per_class": map_per_class_values,
                        f"{prefix}mar_{max_detection_thresholds[-1]}_per_class": mar_per_class_values,
                    },
                )
    result_dict.update({
        "classes": torch.tensor(
            get_classes(detection_labels=detection_labels, groundtruth_labels=groundtruth_labels), dtype=torch.int32
        )
    })
    return {k: (v.squeeze() if isinstance(v, torch.Tensor) and v.numel() == 1 else v) for k, v in result_dict.items()}
