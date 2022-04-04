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


from typing import Dict, List, Set, Tuple

import torch
from torch import Tensor


def _nested_tuple(nested_list: List) -> Tuple:
    """Construct a nested tuple from a nested list."""
    return tuple(map(_nested_tuple, nested_list)) if isinstance(nested_list, list) else nested_list


def _totuple(t: torch.Tensor) -> Tuple:
    """Convert a tensor into a nested tuple."""
    return _nested_tuple(t.tolist())


def _get_color_areas(img: torch.Tensor) -> Dict[Tuple, torch.Tensor]:
    """Calculate a dictionary {pixel_color: area}."""
    unique_keys, unique_keys_area = torch.unique(img, dim=0, return_counts=True)
    # dictionary indexed by color tuples
    return dict(zip(_totuple(unique_keys), unique_keys_area))


def _is_set_int(value) -> bool:
    """Check wheter value is a `Set[int]`"""
    return isinstance(value, Set) and set(map(type, value)).issubset({int})


def _validate_categories(things: Set[int], stuff: Set[int]):
    if not _is_set_int(things):
        raise ValueError("Expected argument `things` to be of type `Set[int]`")
    if not _is_set_int(stuff):
        raise ValueError("Expected argument `stuff` to be of type `Set[int]`")
    if stuff & things:
        raise ValueError("Expected arguments `things` and `stuffs` to have distinct keys.")


def _validate_inputs(preds: torch.Tensor, target: torch.Tensor) -> None:
    if not isinstance(preds, torch.Tensor):
        raise ValueError("Expected argument `preds` to be of type `torch.Tensor`")
    if not isinstance(target, torch.Tensor):
        raise ValueError("Expected argument `target` to be of type `torch.Tensor`")
    if preds.shape != target.shape:
        raise ValueError("Expected argument `preds` and `target` to have the same shape")
    if preds.dim() != 3 or preds.shape[-1] != 2:
        raise ValueError("Expected argument `preds` to have shape [height, width, 2]")


def _get_void_color(things: Set[int], stuff: Set[int]):
    unused_category_id = 1 + max([0] + list(things) + list(stuff))
    return (unused_category_id, 0)


def _get_category_id_to_continous_id(things: Set[int], stuff: Set[int]):
    # things metrics are stored with a continous id in [0, len(things)[,
    thing_id_to_continuous_id = {thing_id: idx for idx, thing_id in enumerate(things)}
    # stuff metrics are stored with a continous id in [len(things), len(things) + len(stuffs)[
    stuff_id_to_continuous_id = {stuff_id: idx + len(things) for idx, stuff_id in enumerate(stuff)}
    cat_id_to_continuous_id = {}
    cat_id_to_continuous_id.update(thing_id_to_continuous_id)
    cat_id_to_continuous_id.update(stuff_id_to_continuous_id)
    return cat_id_to_continuous_id


def _isin(arr: torch.tensor, values: List) -> torch.Tensor:
    """basic implementation of torch.isin to support pre 0.10 version."""
    return (arr[..., None] == arr.new(values)).any(-1)


def _prepocess_image(
    things: Set[int],
    stuff: Set[int],
    img: torch.Tensor,
    void_color: Tuple[int, int],
    allow_unknown_category: bool,
) -> torch.Tensor:
    # flatten the height*width dimensions
    img = torch.flatten(img, 0, -2)
    stuff_pixels = _isin(img[:, 0], list(stuff))
    things_pixels = _isin(img[:, 0], list(things))
    # reset instance ids of stuffs
    img[stuff_pixels, 1] = 0
    if not allow_unknown_category and not torch.all(things_pixels | stuff_pixels):
        raise ValueError("Unknown categories found in preds")
    # set unknown categories to void color
    img[~(things_pixels | stuff_pixels)] = img.new(void_color)
    return img


def _panoptic_quality_update(
    flatten_preds: torch.Tensor,
    flatten_target: torch.Tensor,
    cat_id_to_continuous_id: Dict[int, int],
    void_color: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Returns stat scores (iou sum, true positives, false positives, false negatives) required
    to compute accuracy.
    """
    device = flatten_preds.device
    n_categories = len(cat_id_to_continuous_id)
    iou_sum = torch.zeros(n_categories, dtype=torch.double, device=device)
    true_positives = torch.zeros(n_categories, dtype=torch.int, device=device)
    false_positives = torch.zeros(n_categories, dtype=torch.int, device=device)
    false_negatives = torch.zeros(n_categories, dtype=torch.int, device=device)

    # calculate the area of each prediction, ground truth and pairwise intersection
    pred_areas = _get_color_areas(flatten_preds)
    target_areas = _get_color_areas(flatten_target)
    # intersection matrix of shape [height, width, 2, 2]
    intersection_matrix = torch.transpose(torch.stack((flatten_preds, flatten_target), -1), -1, -2)
    intersection_areas = _get_color_areas(intersection_matrix)

    # select intersection of things of same category with iou > 0.5
    pred_segment_matched = set()
    target_segment_matched = set()
    for (pred_color, target_color), intersection in intersection_areas.items():
        # test only non void, matching category
        if target_color == void_color:
            continue
        if pred_color[0] != target_color[0]:
            continue
        continuous_id = cat_id_to_continuous_id[pred_color[0]]
        pred_area = pred_areas[pred_color]
        target_area = target_areas[target_color]
        pred_void_area = intersection_areas.get((pred_color, void_color), 0)
        void_target_area = intersection_areas.get((void_color, target_color), 0)
        union = pred_area - pred_void_area + target_area - void_target_area - intersection
        iou = intersection / union

        if iou > 0.5:
            pred_segment_matched.add(pred_color)
            target_segment_matched.add(target_color)
            iou_sum[continuous_id] += iou
            true_positives[continuous_id] += 1

    # count false negative: ground truth but not matched
    # areas that are mostly void in the prediction are ignored
    false_negative_colors = set(target_areas.keys()).difference(target_segment_matched)
    false_negative_colors.discard(void_color)
    for target_color in false_negative_colors:
        void_target_area = intersection_areas.get((void_color, target_color), 0)
        if void_target_area / target_areas[target_color] > 0.5:
            continue
        continuous_id = cat_id_to_continuous_id[target_color[0]]
        false_negatives[continuous_id] += 1

    # count false positive: predicted but not matched
    # areas that are mostly void in the target are ignored
    false_positive_colors = set(pred_areas.keys()).difference(pred_segment_matched)
    false_positive_colors.discard(void_color)
    for pred_color in false_positive_colors:
        pred_void_area = intersection_areas.get((pred_color, void_color), 0)
        if pred_void_area / pred_areas[pred_color] > 0.5:
            continue
        continuous_id = cat_id_to_continuous_id[pred_color[0]]
        false_positives[continuous_id] += 1

    return iou_sum, true_positives, false_positives, false_negatives


def _panoptic_quality_compute(
    things: Set[int],
    stuff: Set[int],
    iou_sum: torch.Tensor,
    true_positives: torch.Tensor,
    false_positives: torch.Tensor,
    false_negatives: torch.Tensor,
) -> Dict:
    # TODO: exclude from mean categories that are never seen ?
    # TODO: per class metrics

    # per category calculation
    denominator = (true_positives + 0.5 * false_positives + 0.5 * false_negatives).double()
    panoptic_quality = torch.where(denominator > 0.0, iou_sum / denominator, 0.0)
    segmentation_quality = torch.where(true_positives > 0.0, iou_sum / true_positives, 0.0)
    recognition_quality = torch.where(denominator > 0.0, true_positives / denominator, 0.0)

    metrics = dict(
        all=dict(
            pq=torch.mean(panoptic_quality),
            rq=torch.mean(recognition_quality),
            sq=torch.mean(segmentation_quality),
            n=len(things) + len(stuff),
        ),
        things=dict(
            pq=torch.mean(panoptic_quality[: len(things)]),
            rq=torch.mean(recognition_quality[: len(things)]),
            sq=torch.mean(segmentation_quality[: len(things)]),
            n=len(things),
        ),
        stuff=dict(
            pq=torch.mean(panoptic_quality[len(things) :]),
            rq=torch.mean(recognition_quality[len(things) :]),
            sq=torch.mean(segmentation_quality[len(things) :]),
            n=len(stuff),
        ),
    )

    return metrics


def panoptic_quality(
    preds: torch.Tensor,
    target: torch.Tensor,
    things: Set[int],
    stuff: Set[int],
    allow_unknown_preds_category: bool = False,
) -> Tensor:
    _validate_categories(things, stuff)
    _validate_inputs(preds, target)
    void_color = _get_void_color(things, stuff)
    cat_id_to_continuous_id = _get_category_id_to_continous_id(things, stuff)
    flatten_preds = _prepocess_image(things, stuff, preds, void_color, allow_unknown_preds_category)
    flatten_target = _prepocess_image(things, stuff, target, void_color, True)
    iou_sum, true_positives, false_positives, false_negatives = _panoptic_quality_update(
        flatten_preds, flatten_target, cat_id_to_continuous_id, void_color
    )
    results = _panoptic_quality_compute(things, stuff, iou_sum, true_positives, false_positives, false_negatives)
    return results["all"]["pq"]
