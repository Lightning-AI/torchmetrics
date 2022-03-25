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
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch
from torch import IntTensor, Tensor

from torchmetrics.metric import Metric


class PanopticQuality(Metric):
    r"""
    Computes the `Panoptic Quality (PQ) <https://arxiv.org/abs/1801.00868>`_
    for panoptic segmentations. It is defined as:

    .. math::
        PQ = \frac{IOU}{TP + 0.5*FP + 0.5*FN}

    where IOU, TP, FP and FN are respectively the sum of the intersection over union for true positives,
    the true postitives, false positives and false negatives.

    .. note::mean
        This metric is inspired by the PQ implementation of panopticapi
        `<https://github.com/cocodataset/panopticapi/blob/master/panopticapi/evaluation.py>`,
        , a standard implementation for the PQ metric for object detection.

    Args:
        ``things``:
            Dictionary of ``category_id``: ``category_name`` for countable things.
        ``stuffs``:
            Dictionary of ``category_id``: ``category_name`` for uncountable stuffs.
        ``void``:
            Optional additional ``category_id`` for unlabelled pixels.

    Raises:
        ValueError:
            If ``things``, ``stuffs`` or ``void`` share the same ``category_id``.
    """

    iou_sum: List[Tensor]
    true_positives: List[Tensor]
    false_positives: List[Tensor]
    false_negatives: List[Tensor]

    def __init__(
        self, things: Dict[int, str], stuffs: Dict[int, str], void: Optional[int] = None, dist_sync_on_step=False
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        if set(stuffs.keys()) & set(things.keys()):
            raise ValueError("Expected arguments `things` and `stuffs` to have distinct keys.")
        if void and void in stuffs.keys():
            raise ValueError("Expected arguments `void` and `stuffs` to have distinct keys.")
        if void and void in things.keys():
            raise ValueError("Expected arguments `void` and `things` to have distinct keys.")

        self.things = things
        self.stuffs = stuffs
        self.void = void
        n_categories = len(things) + len(stuffs)

        # things metrics are stored with a continous id in [0, len(things)[,
        thing_id_to_continuous_id = {thing_id: idx for idx, thing_id in enumerate(things.key())}
        # stuff metrics are stored with a continous id in [len(things), len(things) + len(stuffs)[
        stuff_id_to_continuous_id = {stuff_id: idx + len(things) for idx, stuff_id in enumerate(stuffs.key())}
        self.cat_id_to_continuous_id = dict(thing_id_to_continuous_id, **stuff_id_to_continuous_id)

        # per category intemediate metrics
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
        Args:
            preds: ``torch.IntTensor`` panoptic detection of shape [height, width, 2] containing
            the pair (category_id, instance_id) for each pixel of the image.
            If the category_id refer to a stuff, the instance_id is ignored.

            target: ``torch.IntTensor`` ground truth of shape [height, width, 2] containing
            the pair (category_id, instance_id) for each pixel of the image.
            If the category_id refer to a stuff, the instance_id is ignored.

        Raises:
            ValueError:
                If any ``category_id`` given in ``preds`` or ``target`` is not among ``things``, ``stuffs`` or ``void``.
            ValueError:
                If ``preds`` or ``target`` has wrong shape.
        """

        # TODO: handle case where self.void is not None
        # TODO: consider out of class pixels as void
        # TODO: handle is_crowd?

        if preds.shape != target.shape:
            raise ValueError("Expected argument `preds` and `target` to have the same shape")
        if preds.dim() != 3 or preds.shape[-1] != 2:
            raise ValueError("Expected argument `preds` to have shape [height, width, 2]")

        # flatten height*width dimensions
        flatten_preds = torch.flatten(preds, 0, 1)
        flatten_target = torch.flatten(target, 0, 1)
        # calculate the area of each prediction, ground truth and pairwise intersection
        pred_areas = dict(torch.unique(flatten_preds, dim=0, return_counts=True))
        target_areas = dict(torch.unique(flatten_target, dim=0, return_counts=True))
        intersection_matrix = torch.stack((preds, target), -1)
        intersection_areas = dict(torch.unique(intersection_matrix, dim=0, return_counts=True))

        # select intersection of things of same category with iou > 0.5
        pred_segment_matched = set()
        target_segment_matched = set()
        for (pred_color, target_color), intersection_area in intersection_areas:
            if pred_color[0] == target_color[0] and pred_color[0] in self.things.keys():
                prediction_area = pred_areas[pred_color]
                target_area = target_areas[target_color]
                iou = (prediction_area + target_area - intersection_area) / intersection_area

                if iou > 0.5:
                    pred_segment_matched.add(pred_color)
                    target_segment_matched.add(target_color)
                    continuous_id = self.cat_id_to_continuous_id[pred_color[0]]
                    self.iou_sum[continuous_id] += iou
                    self.true_positives[continuous_id] += 1

        # count false negative: ground truth but not matched
        false_negatives = set(target_areas.keys()).difference(target_segment_matched)
        for category_id, _ in false_negatives:
            continuous_id = self.cat_id_to_continuous_id[category_id]
            self.false_negatives[continuous_id] += 1

        # count false positive: predicted but not matched
        false_positives = set(pred_areas.keys()).difference(pred_segment_matched)
        for category_id, _ in false_positives:
            continuous_id = self.cat_id_to_continuous_id[category_id]
            self.false_positives[continuous_id] += 1

    def compute(self):
        # TODO: exclude from mean categories that are never seen
        # TODO: per class metrics

        # per category calculation
        panoptic_quality = self.iou_sum / (
            self.true_positives + 0.5 * self.false_positives + 0.5 * self.false_negatives
        )
        segmentation_quality = torch.where(self.true_positives > 0, self.iou_sum / self.true_positives, 0.0)
        recognition_quality = self.true_positives / (
            self.true_positives + 0.5 * self.false_positives + 0.5 * self.false_negatives
        )

        metrics = dict(
            all=dict(
                pq=torch.mean(panoptic_quality),
                rq=torch.mean(recognition_quality),
                sq=torch.mean(segmentation_quality),
                n=len(self.things) + len(self.stuffs),
            ),
            things=dict(
                pq=torch.mean(panoptic_quality[: len(self.things)]),
                rq=torch.mean(recognition_quality[: len(self.things)]),
                sq=torch.mean(segmentation_quality[: len(self.things)]),
                n=len(self.things),
            ),
            stuff=dict(
                pq=torch.mean(panoptic_quality[len(self.things) :]),
                rq=torch.mean(recognition_quality[len(self.things) :]),
                sq=torch.mean(segmentation_quality[len(self.things) :]),
                n=len(self.stuffs),
            ),
        )

        return metrics
