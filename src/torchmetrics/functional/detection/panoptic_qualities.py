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
from typing import Collection

import torch
from torch import Tensor, tensor

from torchmetrics.functional.detection._panoptic_quality_common import (
    _get_category_id_to_continuous_id,
    _get_void_color,
    _panoptic_quality_compute,
    _panoptic_quality_update,
    _parse_categories,
    _prepocess_inputs,
    _validate_inputs,
)


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


def modified_panoptic_quality(
    preds: Tensor,
    target: Tensor,
    things: Collection[int],
    stuffs: Collection[int],
    allow_unknown_preds_category: bool = False,
) -> Tensor:
    r"""Compute `Modified Panoptic Quality`_ for panoptic segmentations.

    The metric was introduced in `Seamless Scene Segmentation paper`_, and is an adaptation of the original
    `Panoptic Quality`_ where the metric for a stuff class is computed as

    .. math::
        PQ^{\dagger}_c = \frac{IOU_c}{|S_c|}

    where IOU_c is the sum of the intersection over union of all matching segments for a given class, and \|S_c| is
    the overall number of segments in the ground truth for that class.

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
        >>> preds = tensor([[[0, 0], [0, 1], [6, 0], [7, 0], [0, 2], [1, 0]]])
        >>> target = tensor([[[0, 1], [0, 0], [6, 0], [7, 0], [6, 0], [255, 0]]])
        >>> modified_panoptic_quality(preds, target, things = {0, 1}, stuffs = {6, 7})
        tensor(0.7667, dtype=torch.float64)
    """
    things, stuffs = _parse_categories(things, stuffs)
    _validate_inputs(preds, target)
    void_color = _get_void_color(things, stuffs)
    cat_id_to_continuous_id = _get_category_id_to_continuous_id(things, stuffs)
    flatten_preds = _prepocess_inputs(things, stuffs, preds, void_color, allow_unknown_preds_category)
    flatten_target = _prepocess_inputs(things, stuffs, target, void_color, True)
    iou_sum, true_positives, false_positives, false_negatives = _panoptic_quality_update(
        flatten_preds,
        flatten_target,
        cat_id_to_continuous_id,
        void_color,
        modified_metric_stuffs=stuffs,
    )
    return _panoptic_quality_compute(iou_sum, true_positives, false_positives, false_negatives)
