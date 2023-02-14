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
import warnings
from typing import Any, Set

import torch
from torch import Tensor

from torchmetrics.functional.detection.panoptic_quality import (
    _get_category_id_to_continuous_id,
    _get_void_color,
    _panoptic_quality_compute,
    _panoptic_quality_update,
    _prepocess_image,
    _validate_categories,
    _validate_inputs,
)
from torchmetrics.metric import Metric


class PanopticQuality(Metric):
    r"""Compute the `Panoptic Quality`_ for panoptic segmentations.

    .. math::
        PQ = \frac{IOU}{TP + 0.5 FP + 0.5 FN}

    where IOU, TP, FP and FN are respectively the sum of the intersection over union for true positives, the number of
    true postitives, false positives and false negatives. This metric is inspired by the PQ implementation of
    panopticapi, a standard implementation for the PQ metric for object detection.

    Args:
        things:
            Set of ``category_id`` for countable things.
        stuffs:
            Set of ``category_id`` for uncountable stuffs.
        allow_unknown_preds_category:
            Bool indication if unknown categories in preds is allowed

    Raises:
        ValueError:
            If ``things``, ``stuffs`` share the same ``category_id``.

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
        >>> panoptic_quality = PanopticQuality(things = {0, 1}, stuffs = {6, 7})
        >>> panoptic_quality(preds, target)
        tensor(0.5463, dtype=torch.float64)
    """
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    iou_sum: Tensor
    true_positives: Tensor
    false_positives: Tensor
    false_negatives: Tensor

    def __init__(
        self,
        things: Set[int],
        stuffs: Set[int],
        allow_unknown_preds_category: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        # todo
        warnings.warn("This is experimental version and are actively working on its stability.")

        _validate_categories(things, stuffs)
        self.things = things
        self.stuffs = stuffs
        self.void_color = _get_void_color(things, stuffs)
        self.cat_id_to_continuous_id = _get_category_id_to_continuous_id(things, stuffs)
        self.allow_unknown_preds_category = allow_unknown_preds_category

        # per category intermediate metrics
        n_categories = len(things) + len(stuffs)
        self.add_state("iou_sum", default=torch.zeros(n_categories, dtype=torch.double), dist_reduce_fx="sum")
        self.add_state("true_positives", default=torch.zeros(n_categories, dtype=torch.int), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.zeros(n_categories, dtype=torch.int), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.zeros(n_categories, dtype=torch.int), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        r"""Update state with predictions and targets.

        Args:
            preds: panoptic detection of shape ``[height, width, 2]`` containing
                the pair ``(category_id, instance_id)`` for each pixel of the image.
                If the ``category_id`` refer to a stuff, the instance_id is ignored.

            target: ground truth of shape ``[height, width, 2]`` containing
                the pair ``(category_id, instance_id)`` for each pixel of the image.
                If the ``category_id`` refer to a stuff, the instance_id is ignored.

        Raises:
            TypeError:
                If ``preds`` or ``target`` is not an ``torch.Tensor``
            ValueError:
                If ``preds`` or ``target`` has different shape.
            ValueError:
                If ``preds`` is not a 3D tensor where the final dimension have size 2
        """
        _validate_inputs(preds, target)
        flatten_preds = _prepocess_image(
            self.things, self.stuffs, preds, self.void_color, self.allow_unknown_preds_category
        )
        flatten_target = _prepocess_image(self.things, self.stuffs, target, self.void_color, True)
        iou_sum, true_positives, false_positives, false_negatives = _panoptic_quality_update(
            flatten_preds, flatten_target, self.cat_id_to_continuous_id, self.void_color
        )
        self.iou_sum += iou_sum
        self.true_positives += true_positives
        self.false_positives += false_positives
        self.false_negatives += false_negatives

    def compute(self) -> Tensor:
        """Computes panoptic quality based on inputs passed in to ``update`` previously."""
        return _panoptic_quality_compute(self.iou_sum, self.true_positives, self.false_positives, self.false_negatives)
