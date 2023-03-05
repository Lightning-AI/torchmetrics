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
from typing import Collection, Dict, Iterator, List, Set, Tuple

import torch
from torch import Tensor

from torchmetrics.utilities import rank_zero_warn

_Color = Tuple[int, int]


def _nested_tuple(nested_list: List) -> Tuple:
    """Construct a nested tuple from a nested list.

    Args:
        nested_list: The nested list to convert to a nested tuple.

    Returns:
        A nested tuple with the same content.
    """
    return tuple(map(_nested_tuple, nested_list)) if isinstance(nested_list, list) else nested_list


def _to_tuple(t: Tensor) -> Tuple:
    """Convert a tensor into a nested tuple.

    Args:
        t: The tensor to convert.

    Returns:
        A nested tuple with the same content.
    """
    return _nested_tuple(t.tolist())


def _get_color_areas(inputs: Tensor) -> Dict[Tuple, Tensor]:
    """Measure the size of each instance.

    Args:
        inputs: the input tensor containing the colored pixels.

    Returns:
        A dictionary specifying the `(category_id, instance_id)` and the corresponding number of occurrences.
    """
    unique_keys, unique_keys_area = torch.unique(inputs, dim=0, return_counts=True)
    # dictionary indexed by color tuples
    return dict(zip(_to_tuple(unique_keys), unique_keys_area))


def _parse_categories(things: Collection[int], stuffs: Collection[int]) -> Tuple[Set[int], Set[int]]:
    """Parse and validate metrics arguments for `things` and `stuff`.

    Args:
        things: All possible IDs for things categories.
        stuffs: All possible IDs for stuff categories.

    Returns:
        things_parsed: A set of unique category IDs for the things categories.
        stuffs_parsed: A set of unique category IDs for the stuffs categories.
    """
    things_parsed = set(things)
    if len(things_parsed) < len(things):
        rank_zero_warn("The provided `things` categories contained duplicates, which have been removed.", UserWarning)
    stuffs_parsed = set(stuffs)
    if len(stuffs_parsed) < len(stuffs):
        rank_zero_warn("The provided `stuffs` categories contained duplicates, which have been removed.", UserWarning)
    if not all(isinstance(val, int) for val in things_parsed):
        raise TypeError(f"Expected argument `things` to contain `int` categories, but got {things}")
    if not all(isinstance(val, int) for val in stuffs_parsed):
        raise TypeError(f"Expected argument `stuffs` to contain `int` categories, but got {stuffs}")
    if things_parsed & stuffs_parsed:
        raise ValueError(
            f"Expected arguments `things` and `stuffs` to have distinct keys, but got {things} and {stuffs}"
        )
    if not (things_parsed | stuffs_parsed):
        raise ValueError("At least one of `things` and `stuffs` must be non-empty.")
    return things_parsed, stuffs_parsed


def _validate_inputs(preds: Tensor, target: torch.Tensor) -> None:
    """Validate the shapes of prediction and target tensors.

    Args:
        preds: the prediction tensor
        target: the target tensor
    """
    if not isinstance(preds, Tensor):
        raise TypeError(f"Expected argument `preds` to be of type `torch.Tensor`, but got {type(preds)}")
    if not isinstance(target, Tensor):
        raise TypeError(f"Expected argument `target` to be of type `torch.Tensor`, but got {type(target)}")
    if preds.shape != target.shape:
        raise ValueError(
            f"Expected argument `preds` and `target` to have the same shape, but got {preds.shape} and {target.shape}"
        )
    if preds.dim() < 3:
        raise ValueError(
            "Expected argument `preds` to have at least one spatial dimension (B, *spatial_dims, 2), "
            f"got {preds.shape}"
        )
    if preds.shape[-1] != 2:
        raise ValueError(
            "Expected argument `preds` to have exactly 2 channels in the last dimension (category, instance), "
            f"got {preds.shape} instead"
        )


def _get_void_color(things: Set[int], stuffs: Set[int]) -> Tuple[int, int]:
    """Get an unused color ID.

    Args:
        things: The set of category IDs for things.
        stuffs: The set of category IDs for stuffs.

    Returns:
        A new color ID that does not belong to things nor stuffs.
    """
    unused_category_id = 1 + max([0] + list(things) + list(stuffs))
    return unused_category_id, 0


def _get_category_id_to_continuous_id(things: Set[int], stuffs: Set[int]) -> Dict[int, int]:
    """Convert original IDs to continuous IDs.

    Args:
        things: All unique IDs for things classes.
        stuffs: All unique IDs for stuff classes.

    Returns:
        A mapping from the original category IDs to continuous IDs (i.e., 0, 1, 2, ...).
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


def _prepocess_inputs(
    things: Set[int],
    stuffs: Set[int],
    inputs: Tensor,
    void_color: Tuple[int, int],
    allow_unknown_category: bool,
) -> Tensor:
    """Preprocesses an input tensor for metric calculation.

    NOTE: The input tensor is assumed to have dimension ordering (B, spatial_dim0, ..., spatial_dim_N, 2).
    Spelled out explicitly, this means (B, num_points, 2) for point clouds, (B, H, W, 2) for images, and so on.

    Args:
        things: All category IDs for things classes.
        stuffs: All category IDs for stuff classes.
        inputs: The input tensor.
        void_color: An additional color that is masked out during metrics calculation.
        allow_unknown_category: If true, unknown category IDs are mapped to "void".
            Otherwise, an exception is raised if they occur.

    Returns:
        The preprocessed input tensor flattened along the spatial dimensions.
    """
    # flatten the spatial dimensions of the input tensor, e.g., (B, H, W, C) -> (B, H*W, C).
    out = inputs.detach().clone()
    out = torch.flatten(out, 1, -2)
    mask_stuffs = _isin(out[:, :, 0], list(stuffs))
    mask_things = _isin(out[:, :, 0], list(things))
    # reset instance IDs of stuffs
    mask_stuffs_instance = torch.stack([torch.zeros_like(mask_stuffs), mask_stuffs], dim=-1)
    out[mask_stuffs_instance] = 0
    if not allow_unknown_category and not torch.all(mask_things | mask_stuffs):
        raise ValueError(f"Unknown categories found: {out[~(mask_things|mask_stuffs)]}")
    # set unknown categories to void color
    out[~(mask_things | mask_stuffs)] = out.new(void_color)
    return out


def _calculate_iou(
    pred_color: _Color,
    target_color: _Color,
    pred_areas: Dict[_Color, float],
    target_areas: Dict[_Color, float],
    intersection_areas: Dict[Tuple[_Color, _Color], float],
    void_color: _Color,
) -> Tensor:
    """Helper function that calculates the IoU from precomputed areas of segments and their intersections.

    Args:
        pred_color: The `(category_id, instance_id)`, or "color", of a predicted segment that is being matched with a
            target segment.
        target_color: The `(category_id, instance_id)`, or "color", of a ground truth segment that is being matched
            with a predicted segment.
        pred_areas: Mapping from colors of the predicted segments to their extents.
        target_areas: Mapping from colors of the ground truth segments to their extents.
        intersection_areas: Mapping from tuples of `(pred_color, target_color)` to their extent.
        void_color: An additional color that is masked out during metrics calculation.

    Returns:
        The calculated IoU as a torch.Tensor containing a single scalar value.
    """
    if pred_color[0] != target_color[0]:
        raise ValueError(
            "Attempting to compute IoU on segments with different category ID: "
            f"pred {pred_color[0]}, target {target_color[0]}"
        )
    if pred_color == void_color:
        raise ValueError("Attempting to compute IoU on a void segment.")
    intersection = intersection_areas[(pred_color, target_color)]
    pred_area = pred_areas[pred_color]
    target_area = target_areas[target_color]
    pred_void_area = intersection_areas.get((pred_color, void_color), 0)
    void_target_area = intersection_areas.get((void_color, target_color), 0)
    union = pred_area - pred_void_area + target_area - void_target_area - intersection
    return intersection / union


def _filter_false_negatives(
    target_areas: Dict[_Color, float],
    target_segment_matched: Set[_Color],
    intersection_areas: Dict[Tuple[_Color, _Color], float],
    void_color: Tuple[int, int],
) -> Iterator[int]:
    """Filter false negative segments and yield their category IDs.

    False negatives occur when a ground truth segment is not matched with a prediction.
    Areas that are mostly void in the prediction are ignored.

    Args:
        target_areas: Mapping from colors of the ground truth segments to their extents.
        target_segment_matched: Set of ground truth segments that have been matched to a prediction.
        intersection_areas: Mapping from tuples of `(pred_color, target_color)` to their extent.
        void_color: An additional color that is masked out during metrics calculation.

    Yields:
        Category IDs of segments that account for false negatives.
    """
    false_negative_colors = set(target_areas) - target_segment_matched
    false_negative_colors.discard(void_color)
    for target_color in false_negative_colors:
        void_target_area = intersection_areas.get((void_color, target_color), 0)
        if void_target_area / target_areas[target_color] <= 0.5:
            yield target_color[0]


def _filter_false_positives(
    pred_areas: Dict[_Color, float],
    pred_segment_matched: Set[_Color],
    intersection_areas: Dict[Tuple[_Color, _Color], float],
    void_color: Tuple[int, int],
) -> Iterator[int]:
    """Filter false positive segments and yield their category IDs.

    False positives occur when a predicted segment is not matched with a corresponding target one.
    Areas that are mostly void in the target are ignored.

    Args:
        pred_areas: Mapping from colors of the predicted segments to their extents.
        pred_segment_matched: Set of predicted segments that have been matched to a ground truth.
        intersection_areas: Mapping from tuples of `(pred_color, target_color)` to their extent.
        void_color: An additional color that is masked out during metrics calculation.

    Yields:
        Category IDs of segments that account for false positives.
    """
    false_positive_colors = set(pred_areas) - pred_segment_matched
    false_positive_colors.discard(void_color)
    for pred_color in false_positive_colors:
        pred_void_area = intersection_areas.get((pred_color, void_color), 0)
        if pred_void_area / pred_areas[pred_color] <= 0.5:
            yield pred_color[0]


def _panoptic_quality_update_sample(
    flatten_preds: Tensor,
    flatten_target: Tensor,
    cat_id_to_continuous_id: Dict[int, int],
    void_color: Tuple[int, int],
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Calculate stat scores required to compute accuracy **for a single sample**.

    Computed scores: iou sum, true positives, false positives, false negatives.

    Args:
        flatten_preds: A flattened prediction tensor referring to a single sample, shape (num_points, 2).
        flatten_target: A flattened target tensor referring to a single sample, shape (num_points, 2).
        cat_id_to_continuous_id: Mapping from original category IDs to continuous IDs
        void_color: an additional, unused color.

    Returns:
        - IOU Sum
        - True positives
        - False positives
        - False negatives.
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
    # intersection matrix of shape [num_pixels, 2, 2]
    intersection_matrix = torch.transpose(torch.stack((flatten_preds, flatten_target), -1), -1, -2)
    intersection_areas = _get_color_areas(intersection_matrix)

    # select intersection of things of same category with iou > 0.5
    pred_segment_matched = set()
    target_segment_matched = set()
    for pred_color, target_color in intersection_areas:
        # test only non void, matching category
        if target_color == void_color:
            continue
        if pred_color[0] != target_color[0]:
            continue
        iou = _calculate_iou(pred_color, target_color, pred_areas, target_areas, intersection_areas, void_color)
        if iou > 0.5:
            pred_segment_matched.add(pred_color)
            target_segment_matched.add(target_color)
            continuous_id = cat_id_to_continuous_id[target_color[0]]
            iou_sum[continuous_id] += iou
            true_positives[continuous_id] += 1

    for cat_id in _filter_false_negatives(target_areas, target_segment_matched, intersection_areas, void_color):
        continuous_id = cat_id_to_continuous_id[cat_id]
        false_negatives[continuous_id] += 1

    for cat_id in _filter_false_positives(pred_areas, pred_segment_matched, intersection_areas, void_color):
        continuous_id = cat_id_to_continuous_id[cat_id]
        false_positives[continuous_id] += 1

    return iou_sum, true_positives, false_positives, false_negatives


def _panoptic_quality_update(
    flatten_preds: Tensor,
    flatten_target: Tensor,
    cat_id_to_continuous_id: Dict[int, int],
    void_color: Tuple[int, int],
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Calculate stat scores required to compute accuracy.

    Computed scores: iou sum, true positives, false positives, false negatives.

    Args:
        flatten_preds: A flattened prediction tensor, shape (B, num_points, 2).
        flatten_target: A flattened target tensor, shape (B, num_points, 2).
        cat_id_to_continuous_id: Mapping from original category IDs to continuous IDs.
        void_color: an additional, unused color.

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

    # Loop over each sample independently: segments must not be matched across frames.
    for flatten_preds_single, flatten_target_single in zip(flatten_preds, flatten_target):
        result = _panoptic_quality_update_sample(
            flatten_preds_single,
            flatten_target_single,
            cat_id_to_continuous_id,
            void_color,
        )
        iou_sum += result[0]
        true_positives += result[1]
        false_positives += result[2]
        false_negatives += result[3]

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
        Panoptic quality as a tensor containing a single scalar.
    """
    # per category calculation
    denominator = (true_positives + 0.5 * false_positives + 0.5 * false_negatives).double()
    panoptic_quality = torch.where(denominator > 0.0, iou_sum / denominator, 0.0)
    # Reduce across categories. TODO: is it useful to have the option of returning per class metrics?
    return torch.mean(panoptic_quality[denominator > 0])


def panoptic_quality(
    preds: Tensor,
    target: Tensor,
    things: Collection[int],
    stuffs: Collection[int],
    allow_unknown_preds_category: bool = False,
) -> Tensor:
    r"""Compute `Panoptic Quality`_ for panoptic segmentations.

    .. math::
        PQ = \frac{IOU}{TP + 0.5 FP + 0.5 FN}

    where IOU, TP, FP and FN are respectively the sum of the intersection over union for true positives, the number of
    true postitives, false positives and false negatives. This metric is inspired by the PQ implementation of
    panopticapi, a standard implementation for the PQ metric for object detection.


    .. note:
        Points in the target tensor that do not map to a known category ID are automatically ignored in the metric
        computation.

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
            Boolean flag to specify if unknown categories in the predictions are to be ignored in the metric
            computation or raise an exception when found.

    Raises:
        ValueError:
            If ``things``, ``stuffs`` have at least one common ``category_id``.
        TypeError:
            If ``things``, ``stuffs`` contain non-integer ``category_id``.
        TypeError:
            If ``preds`` or ``target`` is not an ``torch.Tensor``.
        ValueError:
             If ``preds`` or ``target`` has different shape.
        ValueError:
            If ``preds`` has less than 3 dimensions.
        ValueError:
            If the final dimension of ``preds`` has size != 2.

    Example:
        >>> from torch import tensor
        >>> preds = tensor([[[[6, 0], [0, 0], [6, 0], [6, 0]],
        ...                  [[0, 0], [0, 0], [6, 0], [0, 1]],
        ...                  [[0, 0], [0, 0], [6, 0], [0, 1]],
        ...                  [[0, 0], [7, 0], [6, 0], [1, 0]],
        ...                  [[0, 0], [7, 0], [7, 0], [7, 0]]]])
        >>> target = tensor([[[[6, 0], [0, 1], [6, 0], [0, 1]],
        ...                   [[0, 1], [0, 1], [6, 0], [0, 1]],
        ...                   [[0, 1], [0, 1], [6, 0], [1, 0]],
        ...                   [[0, 1], [7, 0], [1, 0], [1, 0]],
        ...                   [[0, 1], [7, 0], [7, 0], [7, 0]]]])
        >>> panoptic_quality(preds, target, things = {0, 1}, stuffs = {6, 7})
        tensor(0.5463, dtype=torch.float64)
    """
    things, stuffs = _parse_categories(things, stuffs)
    _validate_inputs(preds, target)
    void_color = _get_void_color(things, stuffs)
    cat_id_to_continuous_id = _get_category_id_to_continuous_id(things, stuffs)
    flatten_preds = _prepocess_inputs(things, stuffs, preds, void_color, allow_unknown_preds_category)
    flatten_target = _prepocess_inputs(things, stuffs, target, void_color, True)
    iou_sum, true_positives, false_positives, false_negatives = _panoptic_quality_update(
        flatten_preds, flatten_target, cat_id_to_continuous_id, void_color
    )
    return _panoptic_quality_compute(iou_sum, true_positives, false_positives, false_negatives)
