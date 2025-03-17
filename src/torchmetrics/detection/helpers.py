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
from collections.abc import Sequence
from types import ModuleType
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from lightning_utilities import apply_to_collection
from torch import Tensor

from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.imports import (
    _FASTER_COCO_EVAL_AVAILABLE,
    _PYCOCOTOOLS_AVAILABLE,
)

if not (_PYCOCOTOOLS_AVAILABLE or _FASTER_COCO_EVAL_AVAILABLE):
    __doctest_skip__ = [
        "CocoBackend.tm_to_coco",
        "CocoBackend.coco_to_tm",
    ]


def _input_validator(
    preds: Sequence[dict[str, Tensor]],
    targets: Sequence[dict[str, Tensor]],
    iou_type: Union[Literal["bbox", "segm"], tuple[Literal["bbox", "segm"], ...]] = "bbox",
    ignore_score: bool = False,
) -> None:
    """Ensure the correct input format of `preds` and `targets`."""
    if isinstance(iou_type, str):
        iou_type = (iou_type,)

    name_map = {"bbox": "boxes", "segm": "masks"}
    if any(tp not in name_map for tp in iou_type):
        raise Exception(f"IOU type {iou_type} is not supported")
    item_val_name = [name_map[tp] for tp in iou_type]

    if not isinstance(preds, Sequence):
        raise ValueError(f"Expected argument `preds` to be of type Sequence, but got {preds}")
    if not isinstance(targets, Sequence):
        raise ValueError(f"Expected argument `target` to be of type Sequence, but got {targets}")
    if len(preds) != len(targets):
        raise ValueError(
            f"Expected argument `preds` and `target` to have the same length, but got {len(preds)} and {len(targets)}"
        )

    for k in [*item_val_name, "labels"] + (["scores"] if not ignore_score else []):
        if any(k not in p for p in preds):
            raise ValueError(f"Expected all dicts in `preds` to contain the `{k}` key")

    for k in [*item_val_name, "labels"]:
        if any(k not in p for p in targets):
            raise ValueError(f"Expected all dicts in `target` to contain the `{k}` key")

    for ivn in item_val_name:
        if not all(isinstance(pred[ivn], Tensor) for pred in preds):
            raise ValueError(f"Expected all {ivn} in `preds` to be of type Tensor")
    if not ignore_score and not all(isinstance(pred["scores"], Tensor) for pred in preds):
        raise ValueError("Expected all scores in `preds` to be of type Tensor")
    if not all(isinstance(pred["labels"], Tensor) for pred in preds):
        raise ValueError("Expected all labels in `preds` to be of type Tensor")
    for ivn in item_val_name:
        if not all(isinstance(target[ivn], Tensor) for target in targets):
            raise ValueError(f"Expected all {ivn} in `target` to be of type Tensor")
    if not all(isinstance(target["labels"], Tensor) for target in targets):
        raise ValueError("Expected all labels in `target` to be of type Tensor")

    for i, item in enumerate(targets):
        for ivn in item_val_name:
            if item[ivn].size(0) != item["labels"].size(0):
                raise ValueError(
                    f"Input '{ivn}' and labels of sample {i} in targets have a"
                    f" different length (expected {item[ivn].size(0)} labels, got {item['labels'].size(0)})"
                )
    if ignore_score:
        return
    for i, item in enumerate(preds):
        for ivn in item_val_name:
            if not (item[ivn].size(0) == item["labels"].size(0) == item["scores"].size(0)):
                raise ValueError(
                    f"Input '{ivn}', labels and scores of sample {i} in predictions have a"
                    f" different length (expected {item[ivn].size(0)} labels and scores,"
                    f" got {item['labels'].size(0)} labels and {item['scores'].size(0)})"
                )


def _fix_empty_tensors(boxes: Tensor) -> Tensor:
    """Empty tensors can cause problems in DDP mode, this methods corrects them."""
    if boxes.numel() == 0 and boxes.ndim == 1:
        return boxes.unsqueeze(0)
    return boxes


def _validate_iou_type_arg(
    iou_type: Union[Literal["bbox", "segm"], tuple[Literal["bbox", "segm"], ...]] = "bbox",
) -> tuple[Literal["bbox", "segm"], ...]:
    """Validate that iou type argument is correct."""
    allowed_iou_types = ("segm", "bbox")
    if isinstance(iou_type, str):
        iou_type = (iou_type,)
    if any(tp not in allowed_iou_types for tp in iou_type):
        raise ValueError(
            f"Expected argument `iou_type` to be one of {allowed_iou_types} or a tuple of, but got {iou_type}"
        )
    return iou_type


def _load_coco_backend_tools(backend: Literal["pycocotools", "faster_coco_eval"]) -> tuple[object, object, ModuleType]:
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


class CocoBackend:
    """Backend implementation for COCO-style Mean Average Precision (mAP) calculation.

    This class provides the core functionality for evaluating object detection and instance
    segmentation predictions using the Common Objects in Context (COCO) evaluation protocol.
    It supports both the standard 'pycocotools' and optimized 'faster_coco_eval' backends.

    It's used for calculation of mAP in MeanAveragePrecision class. It's a backend that abstracts
    away the mAP calculation with coco package

    Args:
        backend (str): Either 'pycocotools' or 'faster_coco_eval'

    """

    def __init__(self, backend: Literal["pycocotools", "faster_coco_eval"]) -> None:
        if backend not in ("pycocotools", "faster_coco_eval"):
            raise ValueError(
                f"Expected argument `backend` to be one of ('pycocotools', 'faster_coco_eval') but got {backend}"
            )
        self.backend = backend

    @property
    def coco(self) -> object:
        """Returns the coco module for the given backend."""
        coco, _, _ = _load_coco_backend_tools(self.backend)
        return coco

    @property
    def cocoeval(self) -> object:
        """Returns the coco eval module for the given backend."""
        _, cocoeval, _ = _load_coco_backend_tools(self.backend)
        return cocoeval

    @property
    def mask_utils(self) -> object:
        """Returns the mask utils object for the given backend."""
        _, _, mask_utils = _load_coco_backend_tools(self.backend)
        return mask_utils

    def _get_coco_datasets(
        self,
        groundtruth_labels: List[Tensor],
        groundtruth_box: List[Tensor],
        groundtruth_mask: List[Tensor],
        groundtruth_crowds: List[Tensor],
        groundtruth_area: List[Tensor],
        detection_labels: List[Tensor],
        detection_box: List[Tensor],
        detection_mask: List[Tensor],
        detection_scores: List[Tensor],
        iou_type: Union[Literal["bbox", "segm"], tuple[Literal["bbox", "segm"], ...]] = ("bbox",),
        average: Literal["macro", "micro"] = "micro",
    ) -> tuple[object, object]:
        """Returns the coco datasets for the target and the predictions."""
        if average == "micro":
            # for micro averaging we set everything to be the same class
            groundtruth_labels = apply_to_collection(groundtruth_labels, Tensor, lambda x: torch.zeros_like(x))
            detection_labels = apply_to_collection(detection_labels, Tensor, lambda x: torch.zeros_like(x))

        coco_target, coco_preds = self.coco(), self.coco()  # type: ignore[operator]

        # Equivalent to _get_classes function
        all_labels = (
            torch.cat(detection_labels + groundtruth_labels).unique().cpu().tolist()
            if len(detection_labels) > 0 or len(groundtruth_labels) > 0
            else []
        )
        coco_target.dataset = self._get_coco_format(
            labels=groundtruth_labels,
            boxes=groundtruth_box if len(groundtruth_box) > 0 else None,
            masks=groundtruth_mask if len(groundtruth_mask) > 0 else None,
            crowds=groundtruth_crowds,
            area=groundtruth_area,
            iou_type=iou_type,
            all_labels=all_labels,
            average=average,
        )
        coco_preds.dataset = self._get_coco_format(
            labels=detection_labels,
            boxes=detection_box if len(detection_box) > 0 else None,
            masks=detection_mask if len(detection_mask) > 0 else None,
            scores=detection_scores,
            iou_type=iou_type,
            all_labels=all_labels,
            average=average,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            coco_target.createIndex()
            coco_preds.createIndex()

        return coco_preds, coco_target

    def _coco_stats_to_tensor_dict(
        self, stats: list[float], prefix: str, max_detection_thresholds: list[int]
    ) -> dict[str, Tensor]:
        """Converts the output of COCOeval.stats to a dict of tensors."""
        mdt = max_detection_thresholds
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
        iou_type: Union[Literal["bbox", "segm"], tuple[Literal["bbox", "segm"], ...]] = ("bbox",),
        backend: Literal["pycocotools", "faster_coco_eval"] = "pycocotools",
    ) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
        """Utility function for converting .json coco format files to the input format of the mAP metric.

        The function accepts a file for the predictions and a file for the target in coco format and converts them to
        a list of dictionaries containing the boxes, labels and scores in the input format of mAP metric.

        Args:
            coco_preds: Path to the json file containing the predictions in coco format
            coco_target: Path to the json file containing the targets in coco format
            iou_type: Type of input, either `bbox` for bounding boxes or `segm` for segmentation masks
            backend: Backend to use for the conversion. Either `pycocotools` or `faster_coco_eval`.

        Returns:
            A tuple containing the predictions and targets in the input format of mAP metric. Each element of the
            tuple is a list of dictionaries containing the boxes, labels and scores.

        Example:
            >>> # File formats are defined at https://cocodataset.org/#format-data
            >>> # Example files can be found at
            >>> # https://github.com/cocodataset/cocoapi/tree/master/results
            >>> from torchmetrics.detection import MeanAveragePrecision
            >>> preds, target = MeanAveragePrecision().coco_to_tm(
            ...   "instances_val2014_fakebbox100_results.json",
            ...   "val2014_fake_eval_res.txt.json"
            ...   iou_type="bbox"
            ... )  # doctest: +SKIP

        """
        iou_type = _validate_iou_type_arg(iou_type)
        coco, _, _ = _load_coco_backend_tools(backend)

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

    def tm_to_coco(
        self,
        groundtruth_labels: List[Tensor],
        groundtruth_box: List[Tensor],
        groundtruth_mask: List[Tensor],
        groundtruth_crowds: List[Tensor],
        groundtruth_area: List[Tensor],
        detection_labels: List[Tensor],
        detection_box: List[Tensor],
        detection_mask: List[Tensor],
        detection_scores: List[Tensor],
        name: str = "tm_map_input",
        iou_type: Union[Literal["bbox", "segm"], tuple[Literal["bbox", "segm"], ...]] = ("bbox",),
        average: Literal["macro", "micro"] = "micro",
    ) -> None:
        """Utility function for converting the input for mAP metric to coco format and saving it to a json file.

        This function should be used after calling `.update(...)` or `.forward(...)` on all data that should be written
        to the file, as the input is then internally cached. The function then converts to information to coco format
        and writes it to json files.

        Args:
            groundtruth_labels: List of tensors containing the ground truth labels
            groundtruth_box: List of tensors containing the ground truth bounding boxes
            groundtruth_mask: List of tensors containing the ground truth segmentation masks
            groundtruth_crowds: List of tensors indicating whether ground truth annotations are crowd annotations
            groundtruth_area: List of tensors containing the area of ground truth annotations
            detection_labels: List of tensors containing the predicted labels
            detection_box: List of tensors containing the predicted bounding boxes
            detection_mask: List of tensors containing the predicted segmentation masks
            detection_scores: List of tensors containing the confidence scores for predictions
            name: Name of the output file, which will be appended with "_preds.json" and "_target.json"
            iou_type: Type of IoU calculation to use. Can be either "bbox" for bounding box or "segm" for segmentation
            average: Type of averaging to use. Can be either "macro" or "micro"

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
            >>> metric = MeanAveragePrecision(iou_type="bbox")
            >>> metric.update(preds, target)
            >>> metric.tm_to_coco("tm_map_input")

        """
        all_labels = (
            torch.cat(detection_labels + groundtruth_labels).unique().cpu().tolist()
            if len(detection_labels) > 0 or len(groundtruth_labels) > 0
            else []
        )
        target_dataset = self._get_coco_format(
            labels=groundtruth_labels,
            boxes=groundtruth_box if len(groundtruth_box) > 0 else None,
            masks=groundtruth_mask if len(groundtruth_mask) > 0 else None,
            crowds=groundtruth_crowds,
            area=groundtruth_area,
            all_labels=all_labels,
            iou_type=iou_type,
            average=average,
        )
        preds_dataset = self._get_coco_format(
            labels=detection_labels,
            boxes=detection_box if len(detection_box) > 0 else None,
            masks=detection_mask if len(detection_mask) > 0 else None,
            scores=detection_scores,
            all_labels=all_labels,
            iou_type=iou_type,
            average=average,
        )
        if "segm" in iou_type:
            # the rle masks needs to be decoded to be written to a file
            preds_dataset["annotations"] = apply_to_collection(
                preds_dataset["annotations"], dtype=bytes, function=lambda x: x.decode("utf-8")
            )
            preds_dataset["annotations"] = apply_to_collection(
                preds_dataset["annotations"],
                dtype=np.uint32,
                function=lambda x: int(x),
            )
            target_dataset = apply_to_collection(target_dataset, dtype=bytes, function=lambda x: x.decode("utf-8"))
            target_dataset = apply_to_collection(target_dataset, dtype=np.uint32, function=lambda x: int(x))

        preds_json = json.dumps(preds_dataset["annotations"], indent=4)
        target_json = json.dumps(target_dataset, indent=4)

        with open(f"{name}_preds.json", "w") as f:
            f.write(preds_json)

        with open(f"{name}_target.json", "w") as f:
            f.write(target_json)

    def _get_coco_format(
        self,
        labels: List[Tensor],
        all_labels: List[Tensor],
        boxes: Optional[List[Tensor]] = None,
        masks: Optional[List[Tensor]] = None,
        scores: Optional[List[Tensor]] = None,
        crowds: Optional[List[Tensor]] = None,
        area: Optional[List[Tensor]] = None,
        iou_type: Union[Literal["bbox", "segm"], tuple[Literal["bbox", "segm"], ...]] = ("bbox",),
        average: Literal["macro", "micro"] = "micro",
    ) -> dict:
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
            if "segm" in iou_type and len(image_masks) > 0:
                images[-1]["height"], images[-1]["width"] = image_masks[0][0][0], image_masks[0][0][1]  # type: ignore[assignment]

            for k, image_label in enumerate(image_labels):
                if boxes is not None:
                    image_box = image_boxes[k]
                if masks is not None and len(image_masks) > 0:
                    image_mask = image_masks[k]
                    image_mask = {"size": image_mask[0], "counts": image_mask[1]}

                if "bbox" in iou_type and len(image_box) != 4:
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
                    area_stat = self.mask_utils.area(image_mask) if "segm" in iou_type else image_box[2] * image_box[3]
                    if len(iou_type) > 1:
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

        classes = [{"id": i, "name": str(i)} for i in all_labels] if average != "micro" else [{"id": 0, "name": "0"}]
        return {"images": images, "annotations": annotations, "categories": classes}


def _warning_on_too_many_detections(limit: int) -> None:
    rank_zero_warn(
        f"Encountered more than {limit} detections in a single image. This means that certain detections with the"
        " lowest scores will be ignored, that may have an undesirable impact on performance. Please consider adjusting"
        " the `max_detection_threshold` to suit your use case. To disable this warning, set attribute class"
        " `warn_on_many_detections=False`, after initializing the metric.",
        UserWarning,
    )


def _get_safe_item_values(
    iou_type: Union[Literal["bbox", "segm"], Tuple[Literal["bbox", "segm"], ...]],
    box_format: str,
    max_detection_thresholds: List[int],
    coco_backend: CocoBackend,
    item: dict[str, Any],
    warn: bool = False,
) -> tuple[Optional[Tensor], Optional[tuple]]:
    """Convert and return the boxes or masks from the item depending on the iou_type.

    Args:
        iou_type:
            Type of input to process. Supported types are:
                - "bbox": Process bounding boxes
                - "segm": Process segmentation masks
        box_format:
            Input format of given boxes. Supported formats are:
                - 'xyxy': boxes are represented via corners, x1, y1 being top left and x2, y2 being bottom right.
                - 'xywh': boxes are represented via corner, width and height, x1, y2 being top left, w, h being
                  width and height.
                - 'cxcywh': boxes are represented via centre, width and height, cx, cy being center of box, w, h being
                  width and height.
        max_detection_thresholds:
            List of thresholds on maximum detections per image. Used to determine if warnings should be raised
            when the number of detections exceeds these thresholds.
        coco_backend:
            The COCO evaluation backend class type to use for processing the items.
        item:
            Input dictionary containing the boxes or masks to be processed, along with other detection information.
        warn:
            Whether to warn if the number of boxes or masks exceeds the max_detection_thresholds.
            Default is False.

    Returns:
        A tuple containing processed boxes or masks depending on the iou_type. The first element is the
        tensor representation, and the second element contains additional metadata if applicable.

    """
    from torchvision.ops import box_convert

    output = [None, None]
    if "bbox" in iou_type:
        boxes = _fix_empty_tensors(item["boxes"])
        if boxes.numel() > 0:
            boxes = box_convert(boxes, in_fmt=box_format, out_fmt="xywh")
        output[0] = boxes  # type: ignore[call-overload]
    if "segm" in iou_type:
        masks = []
        for i in item["masks"].cpu().numpy():
            rle = coco_backend.mask_utils.encode(np.asfortranarray(i))
            masks.append((tuple(rle["size"]), rle["counts"]))
        output[1] = tuple(masks)  # type: ignore[call-overload]

    def _valid_output_len(idx: int) -> bool:
        val = output[idx]
        if val is None:
            return False
        return len(val) > max_detection_thresholds[-1]

    if warn and (_valid_output_len(0) or _valid_output_len(1)):
        _warning_on_too_many_detections(max_detection_thresholds[-1])
    return output  # type: ignore[return-value]


def _get_classes(detection_labels: List[Tensor], groundtruth_labels: List[Tensor]) -> List[int]:
    if len(detection_labels) > 0 or len(groundtruth_labels) > 0:
        return torch.cat(detection_labels + groundtruth_labels).unique().cpu().tolist()
    return []


def _calculate_map_with_coco(
    coco_backend: CocoBackend,
    groundtruth_labels: List[Tensor],
    groundtruth_box: List[Tensor],
    groundtruth_mask: List[Tensor],
    groundtruth_crowds: List[Tensor],
    groundtruth_area: List[Tensor],
    detection_labels: List[Tensor],
    detection_box: List[Tensor],
    detection_mask: List[Tensor],
    detection_scores: List[Tensor],
    iou_type: Union[Literal["bbox", "segm"], Tuple[Literal["bbox", "segm"], ...]],
    average: Literal["macro", "micro"],
    iou_thresholds: List[float],
    rec_thresholds: List[float],
    max_detection_thresholds: List[int],
    class_metrics: bool,
    extended_summary: bool,
) -> Dict[str, Tensor]:
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
                    for class_id in _get_classes(
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
            _get_classes(detection_labels=detection_labels, groundtruth_labels=groundtruth_labels), dtype=torch.int32
        )
    })
    return result_dict
