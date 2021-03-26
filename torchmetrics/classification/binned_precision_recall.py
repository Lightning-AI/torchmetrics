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
from typing import Tuple, Union, List

import torch
from torchmetrics.metric import Metric

from torchmetrics.utilities.data import METRIC_EPS, to_onehot


# From Lightning's AveragePrecision metric
def _average_precision_compute(
    precision: torch.Tensor,
    recall: torch.Tensor,
    num_classes: int,
) -> Union[List[torch.Tensor], torch.Tensor]:
    # Return the step function integral
    # The following works because the last entry of precision is
    # guaranteed to be 1, as returned by precision_recall_curve
    if num_classes == 1:
        recall = recall[0, :]
        precision = precision[0, :]
        return -torch.sum((recall[1:] - recall[:-1]) * precision[:-1])

    res = []
    for p, r in zip(precision, recall):
        res.append(-torch.sum((r[1:] - r[:-1]) * p[:-1]))
    return res


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
        # TODO: enable assert after changing testing code in Lightning
        # assert not compute_on_step, "computation on each step is not supported"
        super().__init__(compute_on_step=False, **kwargs)
        self.num_classes = num_classes
        self.num_thresholds = num_thresholds
        thresholds = torch.linspace(0, 1, num_thresholds)
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
        precisions = self.TPs / (self.TPs + self.FPs + METRIC_EPS)
        recalls = self.TPs / (self.TPs + self.FNs + METRIC_EPS)
        return (precisions, recalls, self.thresholds)


class BinnedAveragePrecision(BinnedPrecisionRecallCurve):
    def compute(self) -> Union[List[torch.Tensor], torch.Tensor]:
        precisions, recalls, thresholds = super().compute()
        return _average_precision_compute(precisions, recalls, self.num_classes)


class BinnedRecallAtFixedPrecision(BinnedPrecisionRecallCurve):
    def __init__(
        self,
        num_classes: int,
        min_precision: float,
        num_thresholds: int = 100,
        compute_on_step: bool = False,  # will ignore this
        **kwargs
    ):
        # TODO: enable once https://github.com/PyTorchLightning/metrics/pull/122 lands
        # assert not compute_on_step, "computation on each step is not supported"
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

        thresholds = thresholds.repeat(self.num_classes, 1)
        condition = precisions >= self.min_precision
        recalls_at_p = (
            torch.where(
                condition, recalls, torch.scalar_tensor(0.0, device=condition.device)
            )
            .max(dim=1)
            .values
        )
        thresholds_at_p = (
            torch.where(
                condition, thresholds, torch.scalar_tensor(1e6, device=condition.device, dtype=thresholds.dtype)
            )
            .min(dim=1)
            .values
        )
        return (recalls_at_p, thresholds_at_p)
