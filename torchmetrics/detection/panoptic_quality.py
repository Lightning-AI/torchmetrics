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
    """

    .. note::
        This metric is following the PQ implementation of
        `panopticapi <https://github.com/cocodataset/panopticapi/blob/master/panopticapi/evaluation.py>`_,
        , a standard implementation for the PQ metric for object detection.

    """

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

        # per category intemediate metrics
        self.add_state(
            "intersection_over_union_sum", default=torch.zeros(n_categories, dtype=torch.double), dist_reduce_fx="sum"
        )
        self.add_state("true_positives", default=torch.zeros(n_categories, dtype=torch.int), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.zeros(n_categories, dtype=torch.int), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.zeros(n_categories, dtype=torch.int), dist_reduce_fx="sum")

    def update(
        self,
        prediction_panoptic_image: torch.IntTensor,
        target_panoptic_image: torch.IntTensor,
    ):
        pass

    def compute(self):
        # per category calculation
        panoptic_quality = self.intersection_over_union_sum / (
            self.true_positives + 0.5 * self.false_positives + 0.5 * self.false_negatives
        )
        segmentation_quality = torch.where(
            self.true_positives, self.intersection_over_union_sum / self.true_positives, 0.0
        )
        recognition_quality = self.true_positives / (
            self.true_positives + 0.5 * self.false_positives + 0.5 * self.false_negatives
        )

        mean_panoptic_quality = torch.mean(panoptic_quality)
        mean_segmentation_quality = torch.mean(segmentation_quality)
        mean_recognition_quality = torch.mean(recognition_quality)
        metrics = {}
        """
        metrics = dict(
            All=dict(PQ=, RQ=, SQ=, N=), 
            Things=dict(PQ=, RQ=, SQ=, N=), 
            Stuff=dict(PQ=, RQ=, SQ=, N=), 
            per_class=dict( dict(PQ=, RQ=, SQ=, N=)
        )"""

        return metrics
