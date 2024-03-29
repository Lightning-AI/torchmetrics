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
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from torchmetrics.utilities.checks import _check_same_shape


def _input_validator(preds: Sequence[Tensor], target: Sequence[Tensor]) -> None:
    """Ensure the correct input format of `preds` and `targets`"""
    if not isinstance(preds, Sequence):
        raise ValueError("Expected argument `preds` to be of type Sequence")
    if not isinstance(target, Sequence):
        raise ValueError("Expected argument `target` to be of type Sequence")
    if len(preds) != len(target):
        raise ValueError("Expected argument `preds` and `target` to have the same length")
    for prediction, ground_truth in zip(preds, target):
        _check_same_shape(prediction, ground_truth)


def intersect_and_union(pred_label, label, num_classes, ignore_index, label_map=dict(), reduce_zero_label=False):
    """Calculate Intersection and Union.

    Args:
        pred_label (torch.Tensor):
            Prediction segmentation map.
        label (torch.Tensor):
            Ground truth segmentation map.
        num_classes (int):
            Number of categories.
        ignore_index (int):
            Index that will be ignored in evaluation.
        label_map (dict):
            Mapping old labels to new labels. The parameter will work only when label is str. Default: dict().
        reduce_zero_label (bool):
            Whether ignore zero label. The parameter will work only when label is str. Default: False.

    Returns:
        torch.Tensor:
            The intersection of prediction and ground truth histogram on all classes.
        torch.Tensor:
            The union of prediction and ground truth histogram on all classes.
        torch.Tensor:
            The prediction histogram on all classes.
        torch.Tensor:
            The ground truth histogram on all classes.

    """
    if label_map is not None:
        label_copy = label.clone()
        for old_id, new_id in label_map.items():
            label[label_copy == old_id] = new_id

    if reduce_zero_label:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = label != ignore_index
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_pred_label = torch.histc(pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_label = torch.histc(label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_union = area_pred_label + area_label - area_intersect

    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(preds, target, num_classes, ignore_index, label_map=dict(), reduce_zero_label=False):
    """Calculate Total Intersection and Union.

    Args:
        preds (list[torch.Tensor]):
            List of prediction segmentation maps.
        target (list[torch.Tensor]):
            List of ground truth segmentation maps.
        num_classes (int):
            Number of categories.
        ignore_index (int):
            Index that will be ignored in evaluation.
        label_map (dict):
            Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool):
            Whether ignore zero label. Default: False.

    Returns:
        torch.Tensor:
            The intersection of prediction and ground truth histogram on all classes.
        torch.Tensor:
            The union of prediction and ground truth histogram on all classes.
        torch.Tensor:
            The prediction histogram on all classes.
        torch.Tensor:
            The ground truth histogram on all classes.

    """
    total_area_intersect = torch.zeros((num_classes,), dtype=torch.float64)
    total_area_union = torch.zeros((num_classes,), dtype=torch.float64)
    total_area_pred_label = torch.zeros((num_classes,), dtype=torch.float64)
    total_area_label = torch.zeros((num_classes,), dtype=torch.float64)

    for result, gt_seg_map in zip(preds, target):
        area_intersect, area_union, area_pred_label, area_label = intersect_and_union(
            result, gt_seg_map, num_classes, ignore_index, label_map, reduce_zero_label
        )

        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label

    return total_area_intersect, total_area_union, total_area_pred_label, total_area_label


def _mean_iou_update(
    preds: Tensor,
    target: Tensor,
    num_labels: int,
    ignore_index: bool,
    nan_to_num: Optional[int] = None,
    label_map: Optional[Dict[int, int]] = None,
    reduce_labels: bool = False,
) -> Tuple[Tensor, int]:
    """Updates and returns variables required to compute Mean Intersection over Union.

    Checks for same shape of each element of the ``preds`` and ``target`` lists.

    Args:
        preds (list[torch.Tensor]):
            List of prediction segmentation maps.
        target (list[torch.Tensor]):
            List of ground truth segmentation maps.
        num_classes (int):
            Number of categories.
        ignore_index (int):
            Index that will be ignored in evaluation.
        label_map (dict):
            Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool):
            Whether ignore zero label. Default: False.

    """
    _input_validator(preds, target)

    total_area_intersect, total_area_union, total_area_pred_label, total_area_label = total_intersect_and_union(
        preds, target, num_labels, ignore_index, label_map, reduce_labels
    )

    return total_area_intersect, total_area_union, total_area_pred_label, total_area_label


def _mean_iou_compute(total_area_intersect, total_area_union, total_area_pred_label, total_area_label) -> Tensor:
    """Computes Mean Intersection over Union.

    Args:
        total_area_intersect:
            ...
        total_area_union:
            ...
        total_area_pred_label:
            ...
        total_area_label:
            ...

    Example:
        >>> preds = torch.tensor([0., 1, 2, 3])
        >>> target = torch.tensor([0., 1, 2, 2])
        >>> total_area_intersect, total_area_union, total_area_pred_label, total_area_label = _mean_iou_update(preds, target)
        >>> _mean_iou_compute(total_area_intersect, total_area_union, total_area_pred_label, total_area_label)
        tensor(0.2500)

    """
    iou = total_area_intersect / total_area_union

    mean_iou = torch.nanmean(iou)

    return mean_iou


def mean_iou(
    preds: List[Tensor],
    target: List[Tensor],
    num_labels: int,
    ignore_index: bool,
    nan_to_num: Optional[int] = None,
    label_map: Optional[Dict[int, int]] = None,
    reduce_labels: bool = False,
) -> Tensor:
    """Computes Mean Intersection over Union (mIoU).

    Args:
        preds:
            estimated labels
        target:
            ground truth labels
        num_labels:
            number of labels
        ignore_index:
            index that will be ignored in evaluation
        nan_to_num:
            If specified, NaN values will be replaced by the numbers defined by the user. Default: None.
        label_map:
            Mapping old labels to new labels. Default: None.
        reduce_labels:
            Whether to ignore the zero label and reduce all labels by one. Default: False.

    Return:
        Tensor with mIoU.

    Example:
        >>> from torchmetrics.functional.segmentation import mean_iou
        >>> preds = [torch.tensor([[2,0],[2,3]])]
        >>> target = [torch.tensor([[255,255],[2,3]])]
        >>> mean_iou(preds, target)
        tensor(0.2500)

    """
    total_area_intersect, total_area_union, total_area_pred_label, total_area_label = _mean_iou_update(
        preds, target, num_labels, ignore_index, nan_to_num, label_map, reduce_labels
    )
    return _mean_iou_compute(total_area_intersect, total_area_union, total_area_pred_label, total_area_label)
