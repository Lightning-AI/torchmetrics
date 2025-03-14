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
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import torch
from torch import Tensor
from torchvision.ops import box_convert

from torchmetrics.detection.helpers import _fix_empty_tensors
from torchmetrics.utilities.backends import _load_coco_backend_tools


def mean_average_precision(
    preds: List[Dict[str, Any]],
    target: List[Dict[str, Any]],
    iou_threshold: Optional[float] = None,
    replacement_val: float = 0,
    aggregate: bool = True,
    backend: Literal["pycocotools", "faster_coco_eval"] = "pycocotools",
    iou_type: Literal["bbox", "segm"] = "bbox",
    box_format: Literal["xyxy", "xywh", "cxcywh"] = "xyxy",
    average: Literal["macro", "micro"] = "macro",
    max_detection_thresholds: Optional[List[int]] = None,
) -> Union[Tensor, Dict[str, Tensor]]:
    r"""Compute the mean Average Precision (mAP) between predictions and targets.

    The function uses a COCO backend to compute standard COCO metrics. By default it
    works for bounding-box detection (`iou_type="bbox"`) and expects the inputs to be a list
    of dictionaries per image. Each prediction dictionary should contain:

        - "boxes": a tensor of shape (N, 4)
        - "scores": a tensor of shape (N,)
        - "labels": a tensor of shape (N,)

    Each target dictionary should contain at least:

        - "boxes": a tensor of shape (M, 4)
        - "labels": a tensor of shape (M,)

    Optionally targets can provide:

        - "iscrowd": a tensor of shape (M,)
        - "area": a tensor of shape (M,)

    The functional implementation will automatically assign an "image_id" for each dictionary
    if the key is not provided.

    Args:
        preds: List of prediction dictionaries.
        target: List of target dictionaries.
        iou_threshold: If provided, a single IoU threshold is used. Otherwise, a range
            from 0.50 to 0.95 (step=0.05) is used.
        replacement_val: Value to return if there are no images in the predictions or targets.
        aggregate: If True, returns the aggregated mAP (averaged over IoU thresholds),
            otherwise returns a dictionary of detailed COCO stats.
        backend: Backend to use; either "pycocotools" or "faster_coco_eval".
        iou_type: Type of input to use, either "bbox" or "segm".
        box_format: Format of the input boxes. Can be one of "xyxy", "xywh", or "cxcywh".
        average: Averaging method, either "macro" or "micro". Passed to COCO dataset creation.
        max_detection_thresholds: List of detection count thresholds. If None, defaults to [1, 10, 100].

    Returns:
        If aggregate=True, returns a single tensor (mAP).
        If aggregate=False, returns a dictionary of detailed COCO metrics.

    """
    if len(preds) == 0 or len(target) == 0:
        if aggregate:
            return torch.tensor(replacement_val, dtype=torch.float32)
        keys = [
            "map",
            "map_50",
            "map_75",
            "map_small",
            "map_medium",
            "map_large",
            "mar_1",
            "mar_10",
            "mar_100",
            "mar_small",
            "mar_medium",
            "mar_large",
        ]
        return {k: torch.tensor(replacement_val, dtype=torch.float32) for k in keys}

    if iou_threshold is not None:
        iou_thresholds = [iou_threshold]
    else:
        iou_thresholds = torch.linspace(0.5, 0.95, int(round((0.95 - 0.5) / 0.05)) + 1).tolist()

    if max_detection_thresholds is None:
        max_detection_thresholds = [1, 10, 100]

    rec_thresholds = torch.linspace(0.0, 1.0, 101).tolist()

    for idx, sample in enumerate(preds):
        if "image_id" not in sample:
            sample["image_id"] = idx
        if iou_type == "bbox" and "boxes" in sample:
            sample["boxes"] = _fix_empty_tensors(sample["boxes"])
            sample["boxes"] = box_convert(sample["boxes"], in_fmt=box_format, out_fmt="xywh")

    for idx, sample in enumerate(target):
        if "image_id" not in sample:
            sample["image_id"] = idx
        if iou_type == "bbox" and "boxes" in sample:
            sample["boxes"] = _fix_empty_tensors(sample["boxes"])
            sample["boxes"] = box_convert(sample["boxes"], in_fmt=box_format, out_fmt="xywh")

    def _get_coco_format(data: List[Dict[str, Any]], is_prediction: bool) -> dict:
        images = []
        annotations = []
        annotation_id = 1

        for sample in data:
            image_id = sample["image_id"]
            images.append({"id": image_id})
            labels = sample["labels"].cpu().tolist() if isinstance(sample["labels"], Tensor) else sample["labels"]
            n_instances = len(labels)
            if is_prediction:
                scores = sample["scores"].cpu().tolist() if isinstance(sample["scores"], Tensor) else sample["scores"]
                boxes = sample.get("boxes", None)
                if boxes is not None and isinstance(boxes, Tensor):
                    boxes = boxes.cpu().tolist()
            else:
                boxes = sample.get("boxes", None)
                if boxes is not None and isinstance(boxes, Tensor):
                    boxes = boxes.cpu().tolist()
                crowds = sample.get("iscrowd", None)
                if crowds is not None and isinstance(crowds, Tensor):
                    crowds = crowds.cpu().tolist()
                else:
                    crowds = [0] * n_instances
                areas = sample.get("area", None)
                if areas is not None and isinstance(areas, Tensor):
                    areas = areas.cpu().tolist()

            for i in range(n_instances):
                ann = {"id": annotation_id, "image_id": image_id, "category_id": labels[i]}
                if boxes is not None:
                    ann["bbox"] = boxes[i]
                    # Compute area from bbox if not provided.
                    # Assumes bbox is in "xywh" where [x, y, w, h]
                    if not is_prediction:
                        if areas is not None:
                            ann["area"] = areas[i]
                        else:
                            ann["area"] = boxes[i][2] * boxes[i][3]
                    else:
                        # For predictions, compute area if missing.
                        ann["area"] = boxes[i][2] * boxes[i][3]
                # For predictions, add score; for targets add iscrowd.
                if is_prediction:
                    ann["score"] = scores[i]
                else:
                    ann["iscrowd"] = crowds[i] if crowds is not None else 0
                    # If boxes are not present and area is missing, default to 0.
                    if "area" not in ann:
                        ann["area"] = 0
                annotations.append(ann)
                annotation_id += 1

        # Build the list of categories based on the unique labels found.
        all_labels = set()
        for sample in data:
            sample_labels = (
                sample["labels"].cpu().tolist() if isinstance(sample["labels"], Tensor) else sample["labels"]
            )
            all_labels.update(sample_labels)
        categories = [{"id": int(lbl), "name": str(lbl)} for lbl in sorted(all_labels)]
        return {"images": images, "annotations": annotations, "categories": categories}

    preds_dataset = _get_coco_format(preds, is_prediction=True)
    target_dataset = _get_coco_format(target, is_prediction=False)

    coco, cocoeval, _ = _load_coco_backend_tools(backend)
    coco_preds_obj = coco()
    coco_target_obj = coco()

    coco_preds_obj.dataset = preds_dataset
    coco_target_obj.dataset = target_dataset

    with contextlib.redirect_stdout(io.StringIO()):
        coco_preds_obj.createIndex()
        coco_target_obj.createIndex()

    coco_eval_obj = cocoeval(coco_target_obj, coco_preds_obj, iouType=iou_type)
    coco_eval_obj.params.iouThrs = np.array(iou_thresholds, dtype=np.float64)
    coco_eval_obj.params.recThrs = np.array(rec_thresholds, dtype=np.float64)
    coco_eval_obj.params.maxDets = max_detection_thresholds

    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval_obj.evaluate()
        coco_eval_obj.accumulate()
        coco_eval_obj.summarize()

    stats = coco_eval_obj.stats

    if aggregate:
        return torch.tensor(stats[0], dtype=torch.float32)
    mdt = max_detection_thresholds
    return {
        "map": torch.tensor(stats[0], dtype=torch.float32),
        "map_50": torch.tensor(stats[1], dtype=torch.float32),
        "map_75": torch.tensor(stats[2], dtype=torch.float32),
        "map_small": torch.tensor(stats[3], dtype=torch.float32),
        "map_medium": torch.tensor(stats[4], dtype=torch.float32),
        "map_large": torch.tensor(stats[5], dtype=torch.float32),
        f"mar_{mdt[0]}": torch.tensor(stats[6], dtype=torch.float32),
        f"mar_{mdt[1]}": torch.tensor(stats[7], dtype=torch.float32),
        f"mar_{mdt[2]}": torch.tensor(stats[8], dtype=torch.float32),
        "mar_small": torch.tensor(stats[9], dtype=torch.float32),
        "mar_medium": torch.tensor(stats[10], dtype=torch.float32),
        "mar_large": torch.tensor(stats[11], dtype=torch.float32),
    }
