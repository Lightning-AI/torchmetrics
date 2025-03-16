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
    """Compute mean Average Precision (mAP) and mean Average Recall (mAR) metrics for object detection.

    Args:
        preds:
            A list of dictionaries containing the detection predictions for each image. Each dictionary
            should contain the following keys:
            - 'boxes': Tensor of shape (N, 4) containing predicted bounding boxes
            - 'scores': Tensor of shape (N,) containing confidence scores for each detection
            - 'labels': Tensor of shape (N,) containing predicted class labels
            - 'masks': (optional) Tensor containing segmentation masks if iou_type includes 'segm'
        target:
            A list of dictionaries containing the ground truth annotations for each image. Each dictionary
            should contain the following keys:
            - 'boxes': Tensor of shape (M, 4) containing ground truth bounding boxes
            - 'labels': Tensor of shape (M,) containing ground truth class labels
            - 'masks': (optional) Tensor containing segmentation masks if iou_type includes 'segm'
            - 'area': (optional) Tensor of shape (M,) containing the area of each ground truth box
            - 'iscrowd': (optional) Tensor of shape (M,) indicating whether the instance is a crowd
        box_format:
            Input format of given boxes. Supported formats are:
            - 'xyxy': boxes are represented via corners, x1, y1 being top left and x2, y2 being bottom right.
            - 'xywh': boxes are represented via corner, width and height, x1, y2 being top left, w, h being
              width and height.
            - 'cxcywh': boxes are represented via centre, width and height, cx, cy being center of box, w, h being
              width and height.
            Default is 'xyxy'.
        iou_type:
            Type of input (either masks or bounding-boxes) used for computing IOU. Supported IOU types are
            "bbox" or "segm" or both as a tuple. Default is "bbox".
        iou_thresholds:
            IoU thresholds for evaluation. If set to ``None`` it corresponds to the stepped range ``[0.5,...,0.95]``
            with step ``0.05``. Else provide a list of floats. Default is None.
        rec_thresholds:
            Recall thresholds for evaluation. If set to ``None`` it corresponds to the stepped range ``[0,...,1]``
            with step ``0.01``. Else provide a list of floats. Default is None.
        max_detection_thresholds:
            Thresholds on max detections per image. If set to `None` will use thresholds ``[1, 10, 100]``.
            Else, please provide a list of ints of length 3, which is the only supported length by both backends.
            Default is None.
        class_metrics:
            Option to enable per-class metrics for mAP and mAR_100. Has a performance impact that scales linearly with
            the number of classes in the dataset. Default is False.
        extended_summary:
            Option to enable extended summary with additional metrics including IOU, precision and recall. The output
            dictionary will contain additional metrics. Default is False.
        average:
            Method for averaging scores over labels. Choose between "macro" and "micro". Default is "macro".
        backend:
            Backend to use for the evaluation. Choose between "pycocotools" and "faster_coco_eval".
            Default is "pycocotools".
        warn_on_many_detections:
            Whether to warn if the number of detections exceeds the max_detection_thresholds. Default is True.

    Returns:
        If extended_summary is False, returns a tensor with the mean average precision.
        If extended_summary is True, returns a dictionary with various metrics including:
        - 'map': mean average precision averaged across IoU thresholds
        - 'map_50': mean average precision at IoU threshold 0.5
        - 'map_75': mean average precision at IoU threshold 0.75
        - 'map_small', 'map_medium', 'map_large': mAP for different object sizes
        - 'mar_1', 'mar_10', 'mar_100': mean average recall for different max detection thresholds
        - 'mar_small', 'mar_medium', 'mar_large': mAR for different object sizes
        - 'precision', 'recall', 'scores', 'ious': detailed arrays when extended_summary is True

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
