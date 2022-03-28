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

import numpy as np
from typing import Any, Dict, List
import torch
from torch import Tensor

from torchmetrics.metric import Metric


def _nested_tuple(nested_list: List):
    """Construct a nested tuple from a nested list."""
    return tuple(map(_nested_tuple, nested_list)) if isinstance(nested_list, list) else nested_list


def _totuple(t: torch.Tensor):
    """Convert a tensor into a nested tuple."""
    return _nested_tuple(t.tolist())


def _get_color_areas(img: torch.Tensor):
    """Calculate a dictionary {pixel_color: area}."""
    unique_keys, unique_keys_area = torch.unique(img, dim=0, return_counts=True)
    # dictionary indexed by color tuples
    return dict(zip(_totuple(unique_keys), unique_keys_area))


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

    def __init__(self, things: Dict[int, str], stuff: Dict[int, str], dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        if set(stuff.keys()) & set(things.keys()):
            raise ValueError("Expected arguments `things` and `stuffs` to have distinct keys.")

        self.things = things
        self.stuff = stuff

        # void color to mark unlabelled regions
        void_category_id = 1 + max(max(things), max(stuff))
        self.void_color = (void_category_id, 0)
        n_categories = len(things) + len(stuff)

        # things metrics are stored with a continous id in [0, len(things)[,
        thing_id_to_continuous_id = {thing_id: idx for idx, thing_id in enumerate(things.keys())}
        # stuff metrics are stored with a continous id in [len(things), len(things) + len(stuffs)[
        stuff_id_to_continuous_id = {stuff_id: idx + len(things) for idx, stuff_id in enumerate(stuff.keys())}
        self.cat_id_to_continuous_id = {}
        self.cat_id_to_continuous_id.update(thing_id_to_continuous_id)
        self.cat_id_to_continuous_id.update(stuff_id_to_continuous_id)

        # per category intemediate metrics
        self.add_state("iou_sum", default=torch.zeros(n_categories, dtype=torch.double), dist_reduce_fx="sum")
        self.add_state("true_positives", default=torch.zeros(n_categories, dtype=torch.int), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.zeros(n_categories, dtype=torch.int), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.zeros(n_categories, dtype=torch.int), dist_reduce_fx="sum")

    def _prepocess_image(self, img: torch.Tensor) -> torch.Tensor:
        # flatten the height*width dimensions
        img = torch.flatten(img, 0, -2)
        # torch.isin not present in older version of torch, using numpy instead
        img = img.numpy()
        stuff_pixels = np.isin(img[:, 0], list(self.stuff.keys()))
        things_pixels = np.isin(img[:, 0], list(self.things.keys()))
        # reset instance ids of stuffs
        img[stuff_pixels, 1] = 0
        # set unknown categories to void color
        img[~(things_pixels | stuff_pixels)] = self.void_color
        return torch.tensor(img)

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
                If ``preds`` or ``target`` has wrong shape.
        """

        # TODO: handle group label?

        if preds.shape != target.shape:
            raise ValueError("Expected argument `preds` and `target` to have the same shape")
        if preds.dim() != 3 or preds.shape[-1] != 2:
            raise ValueError("Expected argument `preds` to have shape [height, width, 2]")

        preds = self._prepocess_image(preds)
        target = self._prepocess_image(target)

        # calculate the area of each prediction, ground truth and pairwise intersection
        pred_areas = _get_color_areas(preds)
        target_areas = _get_color_areas(target)
        # intersection matrix of shape [height, width, 2, 2]
        intersection_matrix = torch.transpose(torch.stack((preds, target), -1), -1, -2)
        intersection_areas = _get_color_areas(intersection_matrix)

        # select intersection of things of same category with iou > 0.5
        pred_segment_matched = set()
        target_segment_matched = set()
        for (pred_color, target_color), intersection in intersection_areas.items():
            # test only non void, matching category
            if target_color == self.void_color:
                continue
            if pred_color[0] != target_color[0]:
                continue
            continuous_id = self.cat_id_to_continuous_id[pred_color[0]]
            pred_area = pred_areas[pred_color]
            target_area = target_areas[target_color]
            pred_void_area = intersection_areas.get((pred_color, self.void_color), 0)
            void_target_area = intersection_areas.get((self.void_color, target_color), 0)
            union = pred_area - pred_void_area + target_area - void_target_area - intersection
            iou = intersection / union

            if iou > 0.5:
                pred_segment_matched.add(pred_color)
                target_segment_matched.add(target_color)
                self.iou_sum[continuous_id] += iou
                self.true_positives[continuous_id] += 1

        # count false negative: ground truth but not matched
        false_negatives = set(target_areas.keys()).difference(target_segment_matched)
        false_negatives.discard(self.void_color)
        for target_color in false_negatives:
            continuous_id = self.cat_id_to_continuous_id[target_color[0]]
            self.false_negatives[continuous_id] += 1

        # count false positive: predicted but not matched
        false_positives = set(pred_areas.keys()).difference(pred_segment_matched)
        false_positives.discard(self.void_color)
        for pred_color in false_positives:
            continuous_id = self.cat_id_to_continuous_id[pred_color[0]]
            self.false_positives[continuous_id] += 1

    def compute(self):
        # TODO: exclude from mean categories that are never seen ?
        # TODO: per class metrics

        # per category calculation
        denominator = (self.true_positives + 0.5 * self.false_positives + 0.5 * self.false_negatives).double()
        panoptic_quality = torch.where(denominator > 0.0, self.iou_sum / denominator, 0.0)
        segmentation_quality = torch.where(self.true_positives > 0.0, self.iou_sum / self.true_positives, 0.0)
        recognition_quality = torch.where(denominator > 0.0, self.true_positives / denominator, 0.0)

        metrics = dict(
            all=dict(
                pq=torch.mean(panoptic_quality),
                rq=torch.mean(recognition_quality),
                sq=torch.mean(segmentation_quality),
                n=len(self.things) + len(self.stuff),
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
                n=len(self.stuff),
            ),
        )

        return metrics
