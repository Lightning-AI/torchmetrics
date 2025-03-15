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
import copy
import io
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torchvision.ops import box_convert

from torchmetrics.detection.helpers import CocoBackend, _fix_empty_tensors


def mean_average_precision(
    preds: List[Dict[str, Any]],
    target: List[Dict[str, Any]],
    iou_thresholds: Optional[List[float]] = None,
    aggregate: bool = False,
    backend: Literal["pycocotools", "faster_coco_eval"] = "pycocotools",
    iou_type: Literal["bbox", "segm"] = "bbox",
    box_format: Literal["xyxy", "xywh", "cxcywh"] = "xyxy",
    average: Literal["macro", "micro"] = "macro",
    max_detection_thresholds: Optional[List[int]] = None,
    rec_thresholds: Optional[List[float]] = None,
    class_metrics: bool = False,
    example: bool = False,
) -> Union[Tensor, Dict[str, Tensor]]:
    r"""Compute the mean Average Precision (mAP) between predictions and targets with extra options.

    Args:
        preds: A list of dictionaries for predictions. Each element must include:
            - For `iou_type="bbox"`: "boxes", "scores", "labels" (all as Tensors).
            - For `iou_type="segm"`: "scores", "labels" and optionally "masks".
        target: A list of dictionaries for targets. Each element must include:
            - For `iou_type="bbox"`: "boxes", "labels" (and can include "iscrowd", "area").
            - For `iou_type="segm"`: "labels" and optionally "masks".
        iou_thresholds: IoU thresholds for evaluation. If None, defaults to range [0.5, 0.95] with step 0.05.
        aggregate: If True and no extra info is requested, only return the aggregate mAP.
        backend: Which backend to use for evaluation. Options are "pycocotools" or "faster_coco_eval".
        iou_type: Specify whether the evaluation is for "bbox" (bounding boxes) or "segm" (segmentation).
        box_format: The input box format. Supported are "xyxy", "xywh" and "cxcywh". Default is "xyxy".
        average: Specify "macro" or "micro" averaging for the scores.
        max_detection_thresholds: Thresholds on max detections per image. If None, defaults to [1, 10, 100].
        rec_thresholds: Recall thresholds for evaluation. If None, defaults to linspace from 0.0 to 1.0 with 101 points.
        class_metrics: If True, additionally returns per-class mAP and mAR.
        example: If True, also returns the converted coco format datasets.

    Returns:
        Either a single Tensor (aggregate mAP) or a dictionary with detailed metrics.

    """
    _validate_sequence(preds, "preds")
    _validate_sequence(target, "target")

    if iou_type == "bbox":
        _validate_bbox_preds(preds)
        _validate_bbox_target(target)
    elif iou_type == "segm":
        _validate_segm_preds(preds)
        _validate_segm_target(target)
    else:
        raise ValueError("Unsupported iou_type. Expected one of ['bbox', 'segm'].")

    preds = copy.deepcopy(preds)
    target = copy.deepcopy(target)

    max_detection_thresholds = max_detection_thresholds or [1, 10, 100]
    rec_thresholds = rec_thresholds or torch.linspace(0.0, 1.0, 101).tolist()

    # Handle completely empty data
    if len(preds) == 0 and len(target) == 0:
        return _empty_result(aggregate, class_metrics, example, max_detection_thresholds, default_value=-1.0)

    # Handle the case where only preds are empty
    if len(preds) == 0 or all(len(sample.get("scores", [])) == 0 for sample in preds):
        classes = _get_classes_from_target(target)
        if aggregate and not (class_metrics or example):
            return torch.tensor(0.0, dtype=torch.float32)
        return _build_empty_result_for_empty_preds(classes, max_detection_thresholds)

    if iou_thresholds is None:
        num = int(round((0.95 - 0.5) / 0.05)) + 1
        iou_thresholds = torch.linspace(0.5, 0.95, num).tolist()

    # Prepare the input data: assign image IDs and (if bbox) convert boxes to xywh format
    preds = _prepare_data(preds, iou_type, box_format)
    target = _prepare_data(target, iou_type, box_format)

    # For bbox mode, check if all prediction boxes are empty
    if iou_type == "bbox" and all(
        "boxes" in sample and isinstance(sample["boxes"], Tensor) and sample["boxes"].numel() == 0 for sample in preds
    ):
        return _empty_result(aggregate, class_metrics, example, max_detection_thresholds, default_value=-1.0)

    if average == "micro":
        _apply_micro_average(preds)
        _apply_micro_average(target)

    preds_dataset = _get_coco_format(preds, is_prediction=True, iou_type=iou_type, backend=backend)
    target_dataset = _get_coco_format(target, is_prediction=False, iou_type=iou_type, backend=backend)

    stats = _run_coco_evaluation(
        preds_dataset, target_dataset, iou_thresholds, rec_thresholds, max_detection_thresholds, iou_type, backend
    )

    result_dict = _build_result_dict(stats, max_detection_thresholds)

    unique_classes = _get_unique_classes(preds, target)
    if len(unique_classes) == 1:
        result_dict["classes"] = torch.tensor(unique_classes[0], dtype=torch.int32)
    else:
        result_dict["classes"] = torch.tensor(unique_classes, dtype=torch.int32)

    # per class metric
    if class_metrics:
        map_per_class, mar_per_class = _compute_per_class_metrics(
            unique_classes, preds, target, iou_thresholds, rec_thresholds, max_detection_thresholds, iou_type, backend
        )
        result_dict["map_per_class"] = map_per_class
        result_dict[f"mar_{max_detection_thresholds[-1]}_per_class"] = mar_per_class
    else:
        result_dict["map_per_class"] = torch.tensor(-1.0, dtype=torch.float32)
        result_dict[f"mar_{max_detection_thresholds[-1]}_per_class"] = torch.tensor(-1.0, dtype=torch.float32)

    if example:
        result_dict["coco_preds"] = torch.tensor(0.0)
        result_dict["coco_target"] = torch.tensor(0.0)

    if aggregate and not (class_metrics or example):
        return torch.tensor(stats[0], dtype=torch.float32)
    return result_dict


def _validate_sequence(seq: Sequence[Any], name: str) -> None:
    if not isinstance(seq, Sequence):
        raise ValueError(f"Expected argument `{name}` to be of type Sequence")


def _validate_bbox_preds(preds: List[Dict[str, Any]]) -> None:
    for sample in preds:
        if not isinstance(sample, dict):
            raise ValueError("Each element in `preds` is expected to be a dict")
        for key in ["boxes", "scores", "labels"]:
            if key not in sample:
                raise ValueError(f"Expected key `{key}` in each dict in `preds`")
        for key in ["boxes", "scores", "labels"]:
            if not isinstance(sample[key], Tensor):
                raise ValueError(f"Expected `{key}` in `preds` to be of type Tensor")


def _validate_bbox_target(target: List[Dict[str, Any]]) -> None:
    for sample in target:
        if not isinstance(sample, dict):
            raise ValueError("Each element in `target` is expected to be a dict")
        for key in ["boxes", "labels"]:
            if key not in sample:
                raise ValueError(f"Expected key `{key}` in each dict in `target`")
        for key in ["boxes", "labels"]:
            if not isinstance(sample[key], Tensor):
                raise ValueError(f"Expected `{key}` in `target` to be of type Tensor")


def _validate_segm_preds(preds: List[Dict[str, Any]]) -> None:
    for sample in preds:
        if not isinstance(sample, dict):
            raise ValueError("Each element in `preds` is expected to be a dict")
        for key in ["scores", "labels"]:
            if key not in sample:
                raise ValueError(f"Expected key `{key}` in each dict in `preds`")
        for key in ["scores", "labels"]:
            if not isinstance(sample[key], Tensor):
                raise ValueError(f"Expected `{key}` in `preds` to be of type Tensor")


def _validate_segm_target(target: List[Dict[str, Any]]) -> None:
    for sample in target:
        if not isinstance(sample, dict):
            raise ValueError("Each element in `target` is expected to be a dict")
        if "labels" not in sample:
            raise ValueError("Expected key `labels` in each dict in `target`")
        if not isinstance(sample["labels"], Tensor):
            raise ValueError("Expected `labels` in `target` to be of type Tensor")


def _prepare_data(
    data: List[Dict[str, Any]], iou_type: Literal["bbox", "segm"], box_format: Literal["xyxy", "xywh", "cxcywh"]
) -> List[Dict[str, Any]]:
    """For each sample, set an image id and if using bbox mode, fix empty tensors and convert boxes to xywh."""
    for idx, sample in enumerate(data):
        if "image_id" not in sample:
            sample["image_id"] = idx
        if iou_type == "bbox" and "boxes" in sample:
            sample["boxes"] = _fix_empty_tensors(sample["boxes"])
            sample["boxes"] = box_convert(sample["boxes"], in_fmt=box_format, out_fmt="xywh")
    return data


def _apply_micro_average(data: List[Dict[str, Any]]) -> None:
    """Replace the labels with zeros for micro averaging."""
    for sample in data:
        if "original_labels" not in sample:
            sample["original_labels"] = (
                sample["labels"].clone() if isinstance(sample["labels"], Tensor) else list(sample["labels"])
            )
        if isinstance(sample["labels"], Tensor):
            sample["labels"] = torch.zeros_like(sample["labels"])
        else:
            sample["labels"] = [0] * len(sample["labels"])


def _get_coco_format(
    data: List[Dict[str, Any]],
    is_prediction: bool,
    iou_type: Literal["bbox", "segm"],
    backend: Literal["pycocotools", "faster_coco_eval"],
) -> dict:
    """Convert the list of detection/segmentation samples into a COCO-compatible format."""
    images = []
    annotations = []
    annotation_id = 1
    mask_utils = None

    if iou_type == "segm":
        mask_utils = CocoBackend(backend).mask_utils
        if mask_utils is None:
            raise ValueError(f"Failed to load mask utilities with backend {backend}")

    for sample in data:
        image_id = sample["image_id"]
        image_entry = {"id": image_id}
        # For segmentation, add height/width if masks are available
        if iou_type == "segm" and "masks" in sample:
            if isinstance(sample["masks"], Tensor):
                if sample["masks"].size(0) > 0:
                    h, w = sample["masks"].shape[-2:]
                    image_entry["height"] = h
                    image_entry["width"] = w
                else:
                    image_entry["height"] = 0
                    image_entry["width"] = 0
            else:
                if len(sample["masks"]) > 0:
                    h, w = sample["masks"][0][0]
                    image_entry["height"] = h
                    image_entry["width"] = w
                else:
                    image_entry["height"] = 0
                    image_entry["width"] = 0
        images.append(image_entry)

        labels = sample["labels"].cpu().tolist() if isinstance(sample["labels"], Tensor) else sample["labels"]
        n_instances = len(labels)

        if iou_type != "segm":
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
            if iou_type == "segm":
                masks = sample.get("masks", None)
                if masks is not None:
                    if isinstance(masks, Tensor):
                        if masks.size(0) > 0:
                            mask_i = masks[i]
                            mask_array = np.asfortranarray(mask_i.cpu().numpy())
                            rle = mask_utils.encode(mask_array)
                            counts = rle["counts"]
                            if isinstance(counts, bytes):
                                counts = counts.decode("utf-8")
                            ann["segmentation"] = {"size": rle["size"], "counts": counts}
                            bbox = mask_utils.toBbox(rle).tolist()
                            ann["bbox"] = bbox
                            ann["area"] = mask_utils.area(rle)
                        else:
                            ann["segmentation"] = []
                            ann["bbox"] = [0, 0, 0, 0]
                            ann["area"] = 0
                    else:
                        if len(masks) > 0:
                            size, counts = masks[i]
                            ann["segmentation"] = {"size": size, "counts": counts}
                            if "boxes" in sample:
                                boxes_local = sample["boxes"]
                                if isinstance(boxes_local, Tensor):
                                    boxes_local = boxes_local.cpu().tolist()
                                bbox = boxes_local[i]
                                ann["bbox"] = bbox
                                ann["area"] = bbox[2] * bbox[3]
                            else:
                                ann["bbox"] = [0, 0, size[1], size[0]]
                                ann["area"] = size[0] * size[1]
                        else:
                            ann["segmentation"] = []
                            ann["bbox"] = [0, 0, 0, 0]
                            ann["area"] = 0
                else:
                    ann["segmentation"] = []
                    ann["bbox"] = [0, 0, 0, 0]
                    ann["area"] = 0

                if is_prediction:
                    scores = (
                        sample["scores"].cpu().tolist() if isinstance(sample["scores"], Tensor) else sample["scores"]
                    )
                    ann["score"] = scores[i]
                else:
                    crowds = sample.get("iscrowd", None)
                    if crowds is not None:
                        crowds = crowds.cpu().tolist() if isinstance(crowds, Tensor) else crowds
                    else:
                        crowds = [0] * n_instances
                    ann["iscrowd"] = crowds[i]
            else:
                if "boxes" in sample and sample["boxes"] is not None:
                    boxes_local = sample["boxes"]
                    if isinstance(boxes_local, Tensor):
                        boxes_local = boxes_local.cpu().tolist()
                    ann["bbox"] = boxes_local[i]
                    if not is_prediction:
                        if sample.get("area", None) is not None:
                            areas_local = sample["area"]
                            if isinstance(areas_local, Tensor):
                                areas_local = areas_local.cpu().tolist()
                            ann["area"] = areas_local[i]
                        else:
                            ann["area"] = boxes_local[i][2] * boxes_local[i][3]
                    else:
                        ann["area"] = boxes_local[i][2] * boxes_local[i][3]
                if is_prediction:
                    scores = (
                        sample["scores"].cpu().tolist() if isinstance(sample["scores"], Tensor) else sample["scores"]
                    )
                    ann["score"] = scores[i]
                else:
                    crowds = sample.get("iscrowd", None)
                    if crowds is not None:
                        crowds = crowds.cpu().tolist() if isinstance(crowds, Tensor) else crowds
                    else:
                        crowds = [0] * n_instances
                    ann["iscrowd"] = crowds[i]
                    if "area" not in ann:
                        ann["area"] = 0
            annotations.append(ann)
            annotation_id += 1

    all_labels: Set[int] = set()
    for sample in data:
        sample_labels = sample["labels"].cpu().tolist() if isinstance(sample["labels"], Tensor) else sample["labels"]
        all_labels.update(sample_labels)
    categories = [{"id": int(lbl), "name": str(lbl)} for lbl in sorted(all_labels)]
    return {"images": images, "annotations": annotations, "categories": categories}


def _run_coco_evaluation(
    preds_dataset: dict,
    target_dataset: dict,
    iou_thresholds: List[float],
    rec_thresholds: List[float],
    max_detection_thresholds: List[int],
    iou_type: Literal["bbox", "segm"],
    backend: Literal["pycocotools", "faster_coco_eval"],
) -> List[float]:
    """Load and run the COCO evaluation using the specified backend."""
    coco_backend = CocoBackend(backend)
    coco, cocoeval = coco_backend.coco, coco_backend.cocoeval
    coco_preds_obj = coco()  # type: ignore[operator]
    coco_target_obj = coco()  # type: ignore[operator]
    coco_preds_obj.dataset = preds_dataset
    coco_target_obj.dataset = target_dataset

    with contextlib.redirect_stdout(io.StringIO()):
        coco_preds_obj.createIndex()
        coco_target_obj.createIndex()

    coco_eval_obj = cocoeval(coco_target_obj, coco_preds_obj, iouType=iou_type)  # type: ignore[operator]
    coco_eval_obj.params.iouThrs = np.array(iou_thresholds, dtype=np.float64)
    coco_eval_obj.params.recThrs = np.array(rec_thresholds, dtype=np.float64)
    coco_eval_obj.params.maxDets = max_detection_thresholds

    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval_obj.evaluate()
        coco_eval_obj.accumulate()
        coco_eval_obj.summarize()

    return coco_eval_obj.stats


def _build_result_dict(stats: List[float], max_detection_thresholds: List[int]) -> Dict[str, Tensor]:
    keys = {
        "map": stats[0],
        "map_50": stats[1],
        "map_75": stats[2],
        "map_small": stats[3],
        "map_medium": stats[4],
        "map_large": stats[5],
        f"mar_{max_detection_thresholds[0]}": stats[6],
        f"mar_{max_detection_thresholds[1]}": stats[7],
        f"mar_{max_detection_thresholds[2]}": stats[8],
        "mar_small": stats[9],
        "mar_medium": stats[10],
        "mar_large": stats[11],
    }
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in keys.items()}


def _get_unique_classes(preds: List[Dict[str, Any]], target: List[Dict[str, Any]]) -> List[int]:
    """Return a sorted list of unique classes from both predictions and targets."""
    unique: Set[int] = set()
    for sample in preds:
        labs = sample.get("original_labels", sample["labels"])
        if isinstance(labs, Tensor):
            labs = labs.cpu().tolist()
        unique.update(labs)
    for sample in target:
        labs = sample.get("original_labels", sample["labels"])
        if isinstance(labs, Tensor):
            labs = labs.cpu().tolist()
        unique.update(labs)
    return sorted(unique)


def _compute_per_class_metrics(
    unique_classes: List[int],
    preds: List[Dict[str, Any]],
    target: List[Dict[str, Any]],
    iou_thresholds: List[float],
    rec_thresholds: List[float],
    max_detection_thresholds: List[int],
    iou_type: Literal["bbox", "segm"],
    backend: Literal["pycocotools", "faster_coco_eval"],
) -> Tuple[Tensor, Tensor]:
    """Run COCO evaluation for each class and return per-class mAP and mAR."""
    map_list = []
    mar_list = []

    preds_for_class = []
    for sample in preds:
        sample_copy = copy.deepcopy(sample)
        if "original_labels" in sample_copy:
            sample_copy["labels"] = sample_copy["original_labels"]
        preds_for_class.append(sample_copy)
    target_for_class = []
    for sample in target:
        sample_copy = copy.deepcopy(sample)
        if "original_labels" in sample_copy:
            sample_copy["labels"] = sample_copy["original_labels"]
        target_for_class.append(sample_copy)

    preds_ds_class = _get_coco_format(preds_for_class, is_prediction=True, iou_type=iou_type, backend=backend)
    target_ds_class = _get_coco_format(target_for_class, is_prediction=False, iou_type=iou_type, backend=backend)

    for class_id in unique_classes:
        coco_backend = CocoBackend(backend)
        coco = coco_backend.coco
        cocoeval = coco_backend.cocoeval
        coco_preds_obj = coco()  # type: ignore[operator]
        coco_target_obj = coco()  # type: ignore[operator]
        coco_preds_obj.dataset = preds_ds_class
        coco_target_obj.dataset = target_ds_class
        with contextlib.redirect_stdout(io.StringIO()):
            coco_preds_obj.createIndex()
            coco_target_obj.createIndex()
        coco_eval_obj = cocoeval(coco_target_obj, coco_preds_obj, iouType=iou_type)  # type: ignore[operator]
        coco_eval_obj.params.iouThrs = np.array(iou_thresholds, dtype=np.float64)
        coco_eval_obj.params.recThrs = np.array(rec_thresholds, dtype=np.float64)
        coco_eval_obj.params.maxDets = max_detection_thresholds
        coco_eval_obj.params.catIds = [class_id]
        with contextlib.redirect_stdout(io.StringIO()):
            coco_eval_obj.evaluate()
            coco_eval_obj.accumulate()
            coco_eval_obj.summarize()
        class_stats = coco_eval_obj.stats
        map_list.append(torch.tensor(class_stats[0], dtype=torch.float32))
        mar_list.append(torch.tensor(class_stats[8], dtype=torch.float32))

    return torch.stack(map_list), torch.stack(mar_list)


def _get_classes_from_target(target: List[Dict[str, Any]]) -> Union[int, List[int]]:
    classes: Set[int] = set()
    for sample in target:
        labs = sample["labels"].cpu().tolist() if isinstance(sample["labels"], Tensor) else sample["labels"]
        classes.update(labs)
    classes_list = sorted(classes)
    return classes_list[0] if len(classes_list) == 1 else classes_list


def _build_empty_result_for_empty_preds(
    classes: Union[int, List[int]], max_detection_thresholds: List[int]
) -> Dict[str, Tensor]:
    return {
        "map": torch.tensor(0.0, dtype=torch.float32),
        "map_50": torch.tensor(0.0, dtype=torch.float32),
        "map_75": torch.tensor(0.0, dtype=torch.float32),
        "map_small": torch.tensor(-1.0, dtype=torch.float32),
        "map_medium": torch.tensor(-1.0, dtype=torch.float32),
        "map_large": torch.tensor(0.0, dtype=torch.float32),
        f"mar_{max_detection_thresholds[0]}": torch.tensor(0.0, dtype=torch.float32),
        f"mar_{max_detection_thresholds[1]}": torch.tensor(0.0, dtype=torch.float32),
        f"mar_{max_detection_thresholds[2]}": torch.tensor(0.0, dtype=torch.float32),
        "mar_small": torch.tensor(-1.0, dtype=torch.float32),
        "mar_medium": torch.tensor(-1.0, dtype=torch.float32),
        "mar_large": torch.tensor(0.0, dtype=torch.float32),
        "map_per_class": torch.tensor(-1.0, dtype=torch.float32),
        f"mar_{max_detection_thresholds[0]}_per_class": torch.tensor(0.0, dtype=torch.float32),
        f"mar_{max_detection_thresholds[1]}_per_class": torch.tensor(0.0, dtype=torch.float32),
        f"mar_{max_detection_thresholds[2]}_per_class": torch.tensor(-1.0, dtype=torch.float32),
        "classes": torch.tensor(classes, dtype=torch.int32),
    }


def _empty_result(
    aggregate: bool, class_metrics: bool, example: bool, max_detection_thresholds: List[int], default_value: float
) -> Union[Tensor, Dict[str, Tensor]]:
    keys = [
        "map",
        "map_50",
        "map_75",
        "map_small",
        "map_medium",
        "map_large",
        f"mar_{max_detection_thresholds[0]}",
        f"mar_{max_detection_thresholds[1]}",
        f"mar_{max_detection_thresholds[2]}",
        "mar_small",
        "mar_medium",
        "mar_large",
    ]
    if aggregate and not (class_metrics or example):
        return torch.tensor(default_value, dtype=torch.float32)
    return {k: torch.tensor(default_value, dtype=torch.float32) for k in keys}
