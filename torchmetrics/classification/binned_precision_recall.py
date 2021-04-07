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
from typing import List, Tuple, Union

import torch

from torchmetrics.functional.classification.average_precision import _average_precision_compute_with_precision_recall
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import METRIC_EPS, to_onehot


class BinnedPrecisionRecallCurve(Metric):
    """Returns a tensor of recalls for a fixed precision threshold.
    It is a tensor instead of a single number, because it applies to multi-label inputs.
    """

    TPs: torch.Tensor
    FPs: torch.Tensor
    FNs: torch.Tensor
    thresholds: torch.Tensor

    def __init__(
        self,
        num_classes: int,
        num_thresholds: int = 100,
        compute_on_step: bool = False,  # will ignore this
        **kwargs
    ):
        assert not compute_on_step, "computation on each step is not supported"
        super().__init__(compute_on_step=False, **kwargs)
        self.num_classes = num_classes
        self.num_thresholds = num_thresholds
        thresholds = torch.linspace(0, 1.0 + METRIC_EPS, num_thresholds)
        self.register_buffer("thresholds", thresholds)

        for name in ("TPs", "FPs", "FNs"):
            self.add_state(
                name=name,
                default=torch.zeros(num_classes, num_thresholds),
                dist_reduce_fx="sum",
            )

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Args
            preds: (n_samples, n_classes) tensor
            targets: (n_samples, n_classes) tensor
        """
        # binary case
        if len(preds.shape) == len(targets.shape) == 1:
            preds = preds.reshape(-1, 1)
            targets = targets.reshape(-1, 1)

        if len(preds.shape) == len(targets.shape) + 1:
            targets = to_onehot(targets, num_classes=self.num_classes)

        targets = targets == 1
        # Iterate one threshold at a time to conserve memory
        for i in range(self.num_thresholds):
            predictions = preds >= self.thresholds[i]
            self.TPs[:, i] += (targets & predictions).sum(dim=0)
            self.FPs[:, i] += ((~targets) & (predictions)).sum(dim=0)
            self.FNs[:, i] += ((targets) & (~predictions)).sum(dim=0)

    def compute(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns float tensor of size n_classes"""
        precisions = (self.TPs + METRIC_EPS) / (self.TPs + self.FPs + METRIC_EPS)
        recalls = self.TPs / (self.TPs + self.FNs + METRIC_EPS)
        # Need to guarantee that last precision=1 and recall=0
        precisions = torch.cat([precisions, torch.ones(self.num_classes, 1,
                               dtype=precisions.dtype, device=precisions.device)], dim=1)
        recalls = torch.cat([recalls, torch.zeros(self.num_classes, 1,
                            dtype=recalls.dtype, device=recalls.device)], dim=1)
        thresholds = torch.cat([self.thresholds, torch.ones(1, dtype=recalls.dtype, device=recalls.device)], dim=0)
        if self.num_classes == 1:
            return (precisions[0, :], recalls[0, :], thresholds)
        else:
            return (precisions, recalls, thresholds)


class BinnedAveragePrecision(BinnedPrecisionRecallCurve):
    def compute(self) -> Union[List[torch.Tensor], torch.Tensor]:
        precisions, recalls, _ = super().compute()
        return _average_precision_compute_with_precision_recall(precisions, recalls, self.num_classes)


class BinnedRecallAtFixedPrecision(BinnedPrecisionRecallCurve):
    def __init__(
        self,
        num_classes: int,
        min_precision: float,
        num_thresholds: int = 100,
        compute_on_step: bool = False,  # will ignore this
        **kwargs
    ):
        super().__init__(
            num_classes=num_classes,
            num_thresholds=num_thresholds,
            compute_on_step=compute_on_step,
            **kwargs
        )
        self.min_precision = min_precision

    def compute(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns float tensor of size n_classes"""
        precisions, recalls, thresholds = super().compute()
        condition = precisions >= self.min_precision

        if self.num_classes == 1:
            recall_at_p, index = (
                torch.where(
                    condition, recalls, torch.scalar_tensor(0.0, device=condition.device)
                )
                .max(dim=0)
            )
            if recall_at_p == 0.0:
                return recall_at_p, torch.scalar_tensor(1e6, device=condition.device)
            else:
                return recall_at_p, thresholds[index]

        recalls_at_p, indices = (
            torch.where(
                condition, recalls, torch.scalar_tensor(0.0, device=condition.device)
            )
            .max(dim=1)
        )

        thresholds_at_p = torch.zeros_like(recalls_at_p, device=condition.device, dtype=thresholds.dtype)
        for i in range(self.num_classes):
            if recalls_at_p[i] == 0.0:
                thresholds_at_p[i] = 1e6
            else:
                thresholds_at_p[i] = thresholds[indices[i]]

        return (recalls_at_p, thresholds_at_p)
