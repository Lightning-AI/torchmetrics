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

from typing import Dict, List

import torch
from torch import Tensor

from torchmetrics.functional.detection.panoptic_quality import (
    _get_category_id_to_continous_id,
    _get_void_color,
    _panoptic_quality_compute,
    _panoptic_quality_update,
    _prepocess_image,
    _validate_categories,
    _validate_inputs,
)
from torchmetrics.metric import Metric


class PanopticQuality(Metric):
    r"""
    Computes the `Panoptic Quality (PQ) <https://arxiv.org/abs/1801.00868>`_
    for panoptic segmentations. It is defined as:

    .. math::
        PQ = \frac{IOU}{TP + 0.5*FP + 0.5*FN}

    where IOU, TP, FP and FN are respectively the sum of the intersection over union for true positives,
    the number of true postitives, false positives and false negatives.

    .. note::mean
        This metric is inspired by the PQ implementation of panopticapi
        `<https://github.com/cocodataset/panopticapi/blob/master/panopticapi/evaluation.py>`,
        , a standard implementation for the PQ metric for object detection.

    Args:
        ``things``:
            Dictionary of ``category_id``: ``category_name`` for countable things.
        ``stuffs``:
            Dictionary of ``category_id``: ``category_name`` for uncountable stuffs.

    Raises:
        ValueError:
            If ``things``, ``stuffs`` share the same ``category_id``.
    """

    iou_sum: List[Tensor]
    true_positives: List[Tensor]
    false_positives: List[Tensor]
    false_negatives: List[Tensor]

    def __init__(
        self,
        things: Dict[int, str],
        stuff: Dict[int, str],
        allow_unknown_preds_category: bool = False,
        dist_sync_on_step: bool = False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        _validate_categories(things, stuff)
        self.things = things
        self.stuff = stuff
        self.void_color = _get_void_color(things, stuff)
        self.cat_id_to_continuous_id = _get_category_id_to_continous_id(things, stuff)
        self.allow_unknown_preds_category = allow_unknown_preds_category

        # per category intemediate metrics
        n_categories = len(things) + len(stuff)
        self.add_state("iou_sum", default=torch.zeros(n_categories, dtype=torch.double), dist_reduce_fx="sum")
        self.add_state("true_positives", default=torch.zeros(n_categories, dtype=torch.int), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.zeros(n_categories, dtype=torch.int), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.zeros(n_categories, dtype=torch.int), dist_reduce_fx="sum")

    def update(
        self,
        preds: torch.IntTensor,
        target: torch.IntTensor,
    ):
        r"""
        Update state with predictions and targets.

        Args:
            preds: ``torch.IntTensor`` panoptic detection of shape [height, width, 2] containing
            the pair (category_id, instance_id) for each pixel of the image.
            If the category_id refer to a stuff, the instance_id is ignored.

            target: ``torch.IntTensor`` ground truth of shape [height, width, 2] containing
            the pair (category_id, instance_id) for each pixel of the image.
            If the category_id refer to a stuff, the instance_id is ignored.

        Raises:
            ValueError:
                If ``preds`` or ``target`` has wrong shape.
            ValueError:
                If ``preds`` containts
        """
        _validate_inputs(preds, target)
        flatten_preds = _prepocess_image(
            self.things, self.stuff, preds, self.void_color, self.allow_unknown_preds_category
        )
        flatten_target = _prepocess_image(self.things, self.stuff, target, self.void_color, True)
        iou_sum, true_positives, false_positives, false_negatives = _panoptic_quality_update(
            flatten_preds, flatten_target, self.cat_id_to_continuous_id, self.void_color
        )
        self.iou_sum += iou_sum
        self.true_positives += true_positives
        self.false_positives += false_positives
        self.false_negatives += false_negatives

    def compute(self):
        """Computes panoptic quality based on inputs passed in to ``update`` previously."""
        results = _panoptic_quality_compute(
            self.things, self.stuff, self.iou_sum, self.true_positives, self.false_positives, self.false_negatives
        )
        return results
