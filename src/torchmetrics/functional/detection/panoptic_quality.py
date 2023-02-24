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
from typing import Any, Dict, List, Set, Tuple

import torch
from torch import Tensor


def _nested_tuple(nested_list: List) -> Tuple:
    """Construct a nested tuple from a nested list.

    Args:
        nested_list:  the nested list to convert to a nested tuple

    Returns:
        a nested tuple with the same content.
    """
    return tuple(map(_nested_tuple, nested_list)) if isinstance(nested_list, list) else nested_list


def _to_tuple(t: Tensor) -> Tuple:
    """Convert a tensor into a nested tuple.

    Args:
        t: the tensor to convert

    Returns:
        a nested tuple with the same content.
    """
    return _nested_tuple(t.tolist())


def _get_color_areas(img: Tensor) -> Dict[Tuple, Tensor]:
    """Count all color occurrences.

    Args:
        img: the image tensor containing the colored pixels.

    Returns:
        a dictionary specifying the color value and the corresponding area
    """
    unique_keys, unique_keys_area = torch.unique(img, dim=0, return_counts=True)
    # dictionary indexed by color tuples
    return dict(zip(_to_tuple(unique_keys), unique_keys_area))


def _is_set_int(value: Any) -> bool:
    """Check whether value is a ``Set[int]``.

    Args:
        value: the value to check

    Returns:
        True if the value is a ``Set[int]``, ``False`` otherwise
    """
    return isinstance(value, Set) and set(map(type, value)).issubset({int})


def _validate_categories(things: Set[int], stuffs: Set[int]) -> None:
    """Validate netrics arguments for `things` and `stuff`.

    Args:
        things: All possible IDs for things categories
        stuffs: All possible IDs for stuff categories
    """
    if not _is_set_int(things):
        raise TypeError(f"Expected argument `things` to be of type `Set[int]`, but got {things}")
    if not _is_set_int(stuffs):
        raise TypeError(f"Expected argument `stuffs` to be of type `Set[int]`, but got {stuffs}")
    if stuffs & things:
        raise ValueError(
            f"Expected arguments `things` and `stuffs` to have distinct keys, but got {things} and {stuffs}"
        )


def _validate_inputs(preds: Tensor, target: torch.Tensor) -> None:
    """Validate the shapes of prediction and target tensors.

    Args:
        preds: the prediction tensor
        target: the target tensor
    """
    if not isinstance(preds, torch.Tensor):
        raise TypeError(f"Expected argument `preds` to be of type `torch.Tensor`, but got {type(preds)}")
    if not isinstance(target, torch.Tensor):
        raise TypeError(f"Expected argument `target` to be of type `torch.Tensor`, but got {type(target)}")
    if preds.shape != target.shape:
        raise ValueError(
            f"Expected argument `preds` and `target` to have the same shape, but got {preds.shape} and {target.shape}"
        )
    if preds.dim() != 3 or preds.shape[-1] != 2:
        raise ValueError(f"Expected argument `preds` to have shape [height, width, 2], but got {preds.shape}")


def _get_void_color(things: Set[int], stuffs: Set[int]) -> Tuple[int, int]:
    """Get an unused color ID.

    Args:
        things: All things IDs
        stuffs: All stuff IDs

    Returns:
        A new color ID with 0 occurrences
    """
    unused_category_id = 1 + max([0] + list(things) + list(stuffs))
    return unused_category_id, 0


def _get_category_id_to_continuous_id(things: Set[int], stuffs: Set[int]) -> Dict[int, int]:
    """Convert original IDs to continuous IDs.

    Args:
        things: all unique ids for things classes
        stuffs: all unique ids for stuff classes

    Returns:
        A mapping from the original category IDs to continuous IDs
    """
    # things metrics are stored with a continuous id in [0, len(things)[,
    thing_id_to_continuous_id = {thing_id: idx for idx, thing_id in enumerate(things)}
    # stuff metrics are stored with a continuous id in [len(things), len(things) + len(stuffs)[
    stuff_id_to_continuous_id = {stuff_id: idx + len(things) for idx, stuff_id in enumerate(stuffs)}
    cat_id_to_continuous_id = {}
    cat_id_to_continuous_id.update(thing_id_to_continuous_id)
    cat_id_to_continuous_id.update(stuff_id_to_continuous_id)
    return cat_id_to_continuous_id


def _isin(arr: Tensor, values: List) -> Tensor:
    """Check if all values of an arr are in another array. Implementation of torch.isin to support pre 0.10 version.

    Args:
        arr: the torch tensor to check for availabilities
        values: the values to search the tensor for.

    Returns:
        a bool tensor of the same shape as :param:`arr` indicating for each
        position whether the element of the tensor is in :param:`values`
    """
    return (arr[..., None] == arr.new(values)).any(-1)


def _prepocess_image(
    things: Set[int],
    stuffs: Set[int],
    img: Tensor,
    void_color: Tuple[int, int],
    allow_unknown_category: bool,
) -> Tensor:
    """Preprocesses the image for metric calculation.

    Args:
        things: All category IDs for things classes
        stuffs: All category IDs for stuff classes
        img: the image tensor
        void_color: an additional, unused color
        allow_unknown_category:  whether to allow an 'unknown' category.

    Returns:
        the preprocessed image tensor with combined height and width dimensions.
    """
    # flatten the height*width dimensions
    img = torch.flatten(img, 0, -2)
    stuff_pixels = _isin(img[:, 0], list(stuffs))
    things_pixels = _isin(img[:, 0], list(things))
    # reset instance ids of stuffs
    img[stuff_pixels, 1] = 0
    if not allow_unknown_category and not torch.all(things_pixels | stuff_pixels):
        raise ValueError("Unknown categories found in preds")
    # set unknown categories to void color
    img[~(things_pixels | stuff_pixels)] = img.new(void_color)
    return img


def _panoptic_quality_update(
    flatten_preds: Tensor,
    flatten_target: Tensor,
    cat_id_to_continuous_id: Dict[int, int],
    void_color: Tuple[int, int],
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Calculate stat scores required to compute accuracy.

    Computed scores: iou sum, true positives, false positives, false negatives

    Args:
        flatten_preds: a flattened prediction tensor
        flatten_target: a flattened target tensor
        cat_id_to_continuous_id: mapping from original category IDs to continuous IDs
        void_color: an additional, unused color

    Returns:
        - IOU Sum
        - True positives
        - False positives
        - False negatives
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
    iou_sum: Tensor,
    true_positives: Tensor,
    false_positives: Tensor,
    false_negatives: Tensor,
) -> Tensor:
    """Compute the final panoptic quality from interim values.

    Args:
        iou_sum: the iou sum from the update step
        true_positives: the TP value from the update step
        false_positives: the FP value from the update step
        false_negatives: the FN value from the update step

    Returns:
        panoptic quality
    """
    # TODO: exclude from mean categories that are never seen ?
    # TODO: per class metrics

    # per category calculation
    denominator = (true_positives + 0.5 * false_positives + 0.5 * false_negatives).double()
    panoptic_quality = torch.where(denominator > 0.0, iou_sum / denominator, 0.0)
    return torch.mean(panoptic_quality)


def panoptic_quality(
    preds: Tensor,
    target: Tensor,
    things: Set[int],
    stuffs: Set[int],
    allow_unknown_preds_category: bool = False,
) -> Tensor:
    r"""Compute `Panoptic Quality`_ for panoptic segmentations.

    .. math::
        PQ = \frac{IOU}{TP + 0.5 FP + 0.5 FN}

    where IOU, TP, FP and FN are respectively the sum of the intersection over union for true positives, the number of
    true postitives, false positives and false negatives. This metric is inspired by the PQ implementation of
    panopticapi, a standard implementation for the PQ metric for object detection.

    Args:
        preds:
            torch tensor with panoptic detection of shape [height, width, 2] containing the pair
            (category_id, instance_id) for each pixel of the image. If the category_id refer to a stuff, the
            instance_id is ignored.
        target:
            torch tensor with ground truth of shape [height, width, 2] containing the pair (category_id, instance_id)
            for each pixel of the image. If the category_id refer to a stuff, the instance_id is ignored.
        things:
            Set of ``category_id`` for countable things.
        stuffs:
            Set of ``category_id`` for uncountable stuffs.
        allow_unknown_preds_category:
            Bool indication if unknown categories in preds is allowed

    Raises:
        ValueError:
            If ``things``, ``stuffs`` share the same ``category_id``.
        TypeError:
            If ``preds`` or ``target`` is not an ``torch.Tensor``
        ValueError:
             If ``preds`` or ``target`` has different shape.
        ValueError:
            If ``preds`` is not a 3D tensor where the final dimension have size 2

    Example:
        >>> from torch import tensor
        >>> preds = tensor([[[6, 0], [0, 0], [6, 0], [6, 0]],
        ...                 [[0, 0], [0, 0], [6, 0], [0, 1]],
        ...                 [[0, 0], [0, 0], [6, 0], [0, 1]],
        ...                 [[0, 0], [7, 0], [6, 0], [1, 0]],
        ...                 [[0, 0], [7, 0], [7, 0], [7, 0]]])
        >>> target = tensor([[[6, 0], [0, 1], [6, 0], [0, 1]],
        ...                  [[0, 1], [0, 1], [6, 0], [0, 1]],
        ...                  [[0, 1], [0, 1], [6, 0], [1, 0]],
        ...                  [[0, 1], [7, 0], [1, 0], [1, 0]],
        ...                  [[0, 1], [7, 0], [7, 0], [7, 0]]])
        >>> panoptic_quality(preds, target, things = {0, 1}, stuffs = {6, 7})
        tensor(0.5463, dtype=torch.float64)
    """
    _validate_categories(things, stuffs)
    _validate_inputs(preds, target)
    void_color = _get_void_color(things, stuffs)
    cat_id_to_continuous_id = _get_category_id_to_continuous_id(things, stuffs)
    flatten_preds = _prepocess_image(things, stuffs, preds, void_color, allow_unknown_preds_category)
    flatten_target = _prepocess_image(things, stuffs, target, void_color, True)
    iou_sum, true_positives, false_positives, false_negatives = _panoptic_quality_update(
        flatten_preds, flatten_target, cat_id_to_continuous_id, void_color
    )
    return _panoptic_quality_compute(iou_sum, true_positives, false_positives, false_negatives)
