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

import torch
from torch import Tensor

__all__ = ["mean_average_precision_3d"]


def _convert_3d_boxes_to_corners(boxes: Tensor, box_format: Literal["xyzwhd", "xyzxyz"]) -> Tensor:
    """Convert 3D boxes to the ``(xmin, ymin, zmin, xmax, ymax, zmax)`` corner format."""
    if box_format == "xyzxyz":
        return boxes
    if box_format == "xyzwhd":
        cx, cy, cz, w, h, d = boxes.unbind(-1)
        return torch.stack(
            [cx - w / 2, cy - h / 2, cz - d / 2, cx + w / 2, cy + h / 2, cz + d / 2],
            dim=-1,
        )
    raise ValueError(f"Expected argument `box_format` to be one of ('xyzwhd', 'xyzxyz') but got {box_format}")


def _volume_3d(boxes: Tensor) -> Tensor:
    """Compute the volume of a set of axis-aligned 3D boxes given in corner format."""
    dx = (boxes[:, 3] - boxes[:, 0]).clamp(min=0)
    dy = (boxes[:, 4] - boxes[:, 1]).clamp(min=0)
    dz = (boxes[:, 5] - boxes[:, 2]).clamp(min=0)
    return dx * dy * dz


def _pairwise_iou_3d(preds: Tensor, target: Tensor) -> Tensor:
    """Compute the pairwise 3D IoU between two sets of axis-aligned boxes given in corner format."""
    if preds.numel() == 0 or target.numel() == 0:
        return torch.zeros((preds.shape[0], target.shape[0]), dtype=torch.float32)

    lt = torch.max(preds[:, None, :3], target[None, :, :3])
    rb = torch.min(preds[:, None, 3:], target[None, :, 3:])
    whd = (rb - lt).clamp(min=0)
    intersection = whd[..., 0] * whd[..., 1] * whd[..., 2]

    vol_preds = _volume_3d(preds)
    vol_target = _volume_3d(target)
    union = vol_preds[:, None] + vol_target[None, :] - intersection
    return torch.where(union > 0, intersection / union, torch.zeros_like(union))


def _compute_average_precision(recall: Tensor, precision: Tensor) -> Tensor:
    """Compute the 101-point interpolated average precision, following the COCO evaluation protocol."""
    recall_thresholds = torch.linspace(0, 1, 101)
    if recall.numel() == 0:
        return torch.tensor(0.0)

    # precision envelope: precision[i] = max(precision[i:])
    envelope = precision.flip(0).cummax(0).values.flip(0)
    indices = torch.searchsorted(recall, recall_thresholds, right=False)
    interpolated = torch.zeros_like(recall_thresholds)
    valid = indices < recall.numel()
    interpolated[valid] = envelope[indices[valid]]
    return interpolated.mean()


def _evaluate_class(
    preds_per_image: List[Tensor],
    scores_per_image: List[Tensor],
    target_per_image: List[Tensor],
    iou_thresholds: List[float],
    max_detection_thresholds: List[int],
) -> Dict[str, Any]:
    """Evaluate a single class across all images, returning AP per IoU threshold and AR per max-detection threshold."""
    num_gt = sum(t.shape[0] for t in target_per_image)

    max_det = max(max_detection_thresholds)
    flat_preds, flat_scores, flat_image_idx = [], [], []
    for image_idx, (boxes, scores) in enumerate(zip(preds_per_image, scores_per_image)):
        if boxes.numel() == 0:
            continue
        order = torch.argsort(scores, descending=True, stable=True)[:max_det]
        flat_preds.append(boxes[order])
        flat_scores.append(scores[order])
        flat_image_idx.extend([image_idx] * order.numel())

    average_precisions = torch.zeros(len(iou_thresholds))
    average_recalls = {mdt: torch.tensor(0.0) for mdt in max_detection_thresholds}

    if num_gt == 0 or len(flat_preds) == 0:
        return {"ap": average_precisions, "ar": average_recalls}

    all_preds = torch.cat(flat_preds, dim=0)
    all_scores = torch.cat(flat_scores, dim=0)
    all_image_idx = torch.tensor(flat_image_idx)

    sort_order = torch.argsort(all_scores, descending=True, stable=True)
    all_preds = all_preds[sort_order]
    all_image_idx = all_image_idx[sort_order]

    ious = [
        _pairwise_iou_3d(all_preds[all_image_idx == image_idx], target_per_image[image_idx])
        if target_per_image[image_idx].numel() > 0
        else None
        for image_idx in range(len(target_per_image))
    ]

    for mdt in max_detection_thresholds:
        matched = [torch.zeros(t.shape[0], dtype=torch.bool) for t in target_per_image]
        per_image_count = dict.fromkeys(range(len(target_per_image)), 0)
        per_image_cursor = dict.fromkeys(range(len(target_per_image)), 0)
        for i in range(all_preds.shape[0]):
            image_idx = int(all_image_idx[i])
            iou_row = ious[image_idx]
            if iou_row is None:
                continue
            pred_cursor = per_image_cursor[image_idx]
            per_image_cursor[image_idx] += 1
            if per_image_count[image_idx] >= mdt:
                continue
            per_image_count[image_idx] += 1
            row = iou_row[pred_cursor].clone()
            row[matched[image_idx]] = -1
            if row.numel() == 0:
                continue
            best_gt = int(row.argmax())
            if row[best_gt] >= min(iou_thresholds):
                matched[image_idx][best_gt] = True
        recall_value = float(sum(m.sum().item() for m in matched)) / num_gt
        average_recalls[mdt] = torch.tensor(recall_value)

    for t_idx, iou_threshold in enumerate(iou_thresholds):
        matched = [torch.zeros(t.shape[0], dtype=torch.bool) for t in target_per_image]
        per_image_cursor = dict.fromkeys(range(len(target_per_image)), 0)
        tp = torch.zeros(all_preds.shape[0])
        fp = torch.zeros(all_preds.shape[0])
        for i in range(all_preds.shape[0]):
            image_idx = int(all_image_idx[i])
            iou_row = ious[image_idx]
            if iou_row is None:
                fp[i] = 1
                continue
            pred_cursor = per_image_cursor[image_idx]
            per_image_cursor[image_idx] += 1
            row = iou_row[pred_cursor].clone()
            row[matched[image_idx]] = -1
            if row.numel() == 0 or row.max() < iou_threshold:
                fp[i] = 1
                continue
            best_gt = int(row.argmax())
            matched[image_idx][best_gt] = True
            tp[i] = 1

        cum_tp = torch.cumsum(tp, dim=0)
        cum_fp = torch.cumsum(fp, dim=0)
        recall = cum_tp / num_gt
        precision = cum_tp / torch.clamp(cum_tp + cum_fp, min=1)
        average_precisions[t_idx] = _compute_average_precision(recall, precision)

    return {"ap": average_precisions, "ar": average_recalls}


def mean_average_precision_3d(
    preds: List[Dict[str, Any]],
    target: List[Dict[str, Any]],
    box_format: Literal["xyzwhd", "xyzxyz"] = "xyzwhd",
    iou_thresholds: Optional[List[float]] = None,
    rec_thresholds: Optional[List[float]] = None,
    max_detection_thresholds: Optional[List[int]] = None,
    class_metrics: bool = False,
) -> Dict[str, Tensor]:
    r"""Compute the mean average precision (mAP) and mean average recall (mAR) for 3D object detection.

    This is the 3D counterpart of :func:`~torchmetrics.functional.detection.mean_average_precision`. Instead of 2D
    bounding boxes it operates on axis-aligned 3D bounding boxes, e.g. as produced by detectors operating on
    point-clouds or voxel grids. Because axis-aligned 3D boxes are used, this implementation does not depend on
    ``pycocotools``/``faster_coco_eval`` and instead computes the intersection-over-union and average precision
    directly.

    Args:
        preds: A list of dictionaries, each containing the keys ``boxes`` (a ``(num_boxes, 6)`` tensor in the format
            given by ``box_format``), ``scores`` (a ``(num_boxes,)`` tensor) and ``labels`` (a ``(num_boxes,)``
            tensor), one dictionary per sample/point-cloud.
        target: A list of dictionaries, each containing the keys ``boxes`` (a ``(num_boxes, 6)`` tensor) and
            ``labels`` (a ``(num_boxes,)`` tensor), one dictionary per sample/point-cloud.
        box_format: Format of the input 3D boxes. Either ``"xyzwhd"`` (center x, y, z and width, height, depth) or
            ``"xyzxyz"`` (min x, y, z and max x, y, z corners).
        iou_thresholds: List of IoU thresholds (default is ``[0.5, 0.55, ..., 0.95]``).
        rec_thresholds: Unused, kept for API parity with the 2D metric. Recall is always evaluated at 101 points.
        max_detection_thresholds: List of maximum detections per sample (default is ``[1, 10, 100]``).
        class_metrics: Whether to compute per-class mAP and mAR metrics.

    Returns:
        dict: A dictionary containing the evaluation metrics:

            - ``map``: Global mean average precision over the defined IoU thresholds.
            - ``map_50``: mAP at IoU=0.50 (``-1`` if 0.5 is not part of ``iou_thresholds``).
            - ``map_75``: mAP at IoU=0.75 (``-1`` if 0.75 is not part of ``iou_thresholds``).
            - ``mar_{max_det}``: Mean average recall for each maximum detection threshold.
            - ``map_per_class``: Mean average precision per observed class (``-1`` if ``class_metrics`` is disabled).
            - ``mar_{max_det}_per_class``: Mean average recall per class at the largest max detection threshold.
            - ``classes``: A tensor listing all observed classes.

    Example::

        >>> from torch import tensor
        >>> from torchmetrics.functional.detection.map_3d import mean_average_precision_3d
        >>> preds = [
        ...   {
        ...     "boxes": tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]]),
        ...     "scores": tensor([0.9]),
        ...     "labels": tensor([0]),
        ...   }
        ... ]
        >>> target = [
        ...   {
        ...     "boxes": tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]]),
        ...     "labels": tensor([0]),
        ...   }
        ... ]
        >>> result = mean_average_precision_3d(preds, target)
        >>> print(f"mAP: {result['map']:.4f}, mAP@0.5: {result['map_50']:.4f}")
        mAP: 1.0000, mAP@0.5: 1.0000

    """
    from torchmetrics.detection.helpers import _input_validator

    _input_validator(preds, target, iou_type="bbox")

    iou_thresholds = iou_thresholds or torch.linspace(0.5, 0.95, round((0.95 - 0.5) / 0.05) + 1).tolist()
    max_detection_thresholds = sorted(max_detection_thresholds or [1, 10, 100])

    pred_boxes = [_convert_3d_boxes_to_corners(item["boxes"], box_format) for item in preds]
    pred_scores = [item["scores"] for item in preds]
    pred_labels = [item["labels"] for item in preds]
    target_boxes = [_convert_3d_boxes_to_corners(item["boxes"], box_format) for item in target]
    target_labels = [item["labels"] for item in target]

    classes = torch.cat(target_labels + pred_labels).unique() if (target_labels or pred_labels) else torch.tensor([])

    per_class_ap: Dict[float, Tensor] = {}
    per_class_ar: Dict[int, Dict[float, Tensor]] = {mdt: {} for mdt in max_detection_thresholds}
    for cls in classes.tolist():
        preds_c = [boxes[labels == cls] for boxes, labels in zip(pred_boxes, pred_labels)]
        scores_c = [scores[labels == cls] for scores, labels in zip(pred_scores, pred_labels)]
        target_c = [boxes[labels == cls] for boxes, labels in zip(target_boxes, target_labels)]
        result = _evaluate_class(preds_c, scores_c, target_c, iou_thresholds, max_detection_thresholds)
        per_class_ap[cls] = result["ap"]
        for mdt in max_detection_thresholds:
            per_class_ar[mdt][cls] = result["ar"][mdt]

    if len(classes) == 0:
        map_per_class = torch.tensor(-1.0)
        map_value = torch.tensor(-1.0)
        map_50 = torch.tensor(-1.0)
        map_75 = torch.tensor(-1.0)
        ar_values = {mdt: torch.tensor(-1.0) for mdt in max_detection_thresholds}
        ar_per_class = {mdt: torch.tensor(-1.0) for mdt in max_detection_thresholds}
    else:
        all_ap = torch.stack(list(per_class_ap.values()))  # (num_classes, num_iou_thresholds)
        map_value = all_ap.mean()
        map_50 = all_ap[:, iou_thresholds.index(0.5)].mean() if 0.5 in iou_thresholds else torch.tensor(-1.0)
        map_75 = all_ap[:, iou_thresholds.index(0.75)].mean() if 0.75 in iou_thresholds else torch.tensor(-1.0)
        map_per_class = all_ap.mean(dim=1) if class_metrics else torch.tensor(-1.0)
        ar_values = {mdt: torch.stack(list(per_class_ar[mdt].values())).mean() for mdt in max_detection_thresholds}
        ar_per_class = {
            mdt: torch.stack(list(per_class_ar[mdt].values())) if class_metrics else torch.tensor(-1.0)
            for mdt in max_detection_thresholds
        }

    result_dict: Dict[str, Tensor] = {
        "map": map_value,
        "map_50": map_50,
        "map_75": map_75,
        "map_per_class": map_per_class,
        "classes": classes.to(torch.int32),
    }
    for mdt in max_detection_thresholds:
        result_dict[f"mar_{mdt}"] = ar_values[mdt]
    result_dict[f"mar_{max_detection_thresholds[-1]}_per_class"] = ar_per_class[max_detection_thresholds[-1]]
    return {k: (v.squeeze() if isinstance(v, torch.Tensor) and v.numel() == 1 else v) for k, v in result_dict.items()}
