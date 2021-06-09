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
from typing import Any, List, Optional, Tuple, Union

import torch
from torch import Tensor

from torchmetrics.functional.classification.average_precision import _average_precision_compute_with_precision_recall
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import METRIC_EPS, to_onehot


def _recall_at_precision(precision: Tensor, recall: Tensor, thresholds: Tensor, min_precision: float):
    try:
        max_recall, max_precision, best_threshold = max((r, p, t) for p, r, t in zip(precision, recall, thresholds)
                                                        if p >= min_precision)
    except ValueError:
        max_recall = torch.tensor(0.0, device=recall.device, dtype=recall.dtype)

    if max_recall == 0.0:
        best_threshold = torch.tensor(1e6, device=thresholds.device, dtype=thresholds.dtype)

    return max_recall, best_threshold


class BinnedPrecisionRecallCurve(Metric):
    """
    Computes precision-recall pairs for different thresholds. Works for both
    binary and multiclass problems. In the case of multiclass, the values will
    be calculated based on a one-vs-the-rest approach.

    Computation is performed in constant-memory by computing precision and recall
    for ``num_thresholds`` buckets/thresholds (evenly distributed between 0 and 1).

    Forward accepts

    - ``preds`` (float tensor): ``(N, ...)`` (binary) or ``(N, C, ...)`` (multiclass) tensor
      with probabilities, where C is the number of classes.

    - ``target`` (long tensor): ``(N, ...)`` or ``(N, C, ...)`` with integer labels

    Args:
        num_classes: integer with number of classes. For binary, set to 1.
        num_thresholds: number of bins used for computation. More bins will lead to more detailed
            curve and accurate estimates, but will be slower and consume more memory. Default 100
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)

    Example (binary case):
        >>> from torchmetrics import BinnedPrecisionRecallCurve
        >>> pred = torch.tensor([0, 0.1, 0.8, 0.4])
        >>> target = torch.tensor([0, 1, 1, 0])
        >>> pr_curve = BinnedPrecisionRecallCurve(num_classes=1, num_thresholds=5)
        >>> precision, recall, thresholds = pr_curve(pred, target)
        >>> precision
        tensor([0.5000, 0.5000, 1.0000, 1.0000, 1.0000, 1.0000])
        >>> recall
        tensor([1.0000, 0.5000, 0.5000, 0.5000, 0.0000, 0.0000])
        >>> thresholds
        tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])

    Example (multiclass case):
        >>> pred = torch.tensor([[0.75, 0.05, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.75, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.75, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.05, 0.75, 0.05]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> pr_curve = BinnedPrecisionRecallCurve(num_classes=5, num_thresholds=3)
        >>> precision, recall, thresholds = pr_curve(pred, target)
        >>> precision   # doctest: +NORMALIZE_WHITESPACE
        [tensor([0.2500, 1.0000, 1.0000, 1.0000]),
        tensor([0.2500, 1.0000, 1.0000, 1.0000]),
        tensor([2.5000e-01, 1.0000e-06, 1.0000e+00, 1.0000e+00]),
        tensor([2.5000e-01, 1.0000e-06, 1.0000e+00, 1.0000e+00]),
        tensor([2.5000e-07, 1.0000e+00, 1.0000e+00, 1.0000e+00])]
        >>> recall   # doctest: +NORMALIZE_WHITESPACE
        [tensor([1.0000, 1.0000, 0.0000, 0.0000]),
        tensor([1.0000, 1.0000, 0.0000, 0.0000]),
        tensor([1.0000, 0.0000, 0.0000, 0.0000]),
        tensor([1.0000, 0.0000, 0.0000, 0.0000]),
        tensor([0., 0., 0., 0.])]
        >>> thresholds   # doctest: +NORMALIZE_WHITESPACE
        [tensor([0.0000, 0.5000, 1.0000]),
        tensor([0.0000, 0.5000, 1.0000]),
        tensor([0.0000, 0.5000, 1.0000]),
        tensor([0.0000, 0.5000, 1.0000]),
        tensor([0.0000, 0.5000, 1.0000])]
    """

    def __init__(
        self,
        num_classes: int,
        num_thresholds: int = 100,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )

        self.num_classes = num_classes
        self.num_thresholds = num_thresholds
        thresholds = torch.linspace(0, 1.0, num_thresholds)
        self.register_buffer("thresholds", thresholds)

        for name in ("TPs", "FPs", "FNs"):
            self.add_state(
                name=name,
                default=torch.zeros(num_classes, num_thresholds, dtype=torch.float32),
                dist_reduce_fx="sum",
            )

    def update(self, preds: Tensor, targets: Tensor) -> None:
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

    def compute(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Returns float tensor of size n_classes"""
        precisions = (self.TPs + METRIC_EPS) / (self.TPs + self.FPs + METRIC_EPS)
        recalls = self.TPs / (self.TPs + self.FNs + METRIC_EPS)

        # Need to guarantee that last precision=1 and recall=0, similar to precision_recall_curve
        t_ones = torch.ones(self.num_classes, 1, dtype=precisions.dtype, device=precisions.device)
        precisions = torch.cat([precisions, t_ones], dim=1)
        t_zeros = torch.zeros(self.num_classes, 1, dtype=recalls.dtype, device=recalls.device)
        recalls = torch.cat([recalls, t_zeros], dim=1)
        if self.num_classes == 1:
            return (precisions[0, :], recalls[0, :], self.thresholds)
        return (list(precisions), list(recalls), [self.thresholds for _ in range(self.num_classes)])


class BinnedAveragePrecision(BinnedPrecisionRecallCurve):
    """
    Computes the average precision score, which summarises the precision recall
    curve into one number. Works for both binary and multiclass problems.
    In the case of multiclass, the values will be calculated based on a one-vs-the-rest approach.

    Computation is performed in constant-memory by computing precision and recall
    for ``num_thresholds`` buckets/thresholds (evenly distributed between 0 and 1).

    Forward accepts

    - ``preds`` (float tensor): ``(N, ...)`` (binary) or ``(N, C, ...)`` (multiclass) tensor
      with probabilities, where C is the number of classes.

    - ``target`` (long tensor): ``(N, ...)`` with integer labels

    Args:
        num_classes: integer with number of classes. Not nessesary to provide
            for binary problems.
        num_thresholds: number of bins used for computation. More bins will lead to more detailed
            curve and accurate estimates, but will be slower and consume more memory. Default 100
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)

    Example (binary case):
        >>> from torchmetrics import BinnedAveragePrecision
        >>> pred = torch.tensor([0, 1, 2, 3])
        >>> target = torch.tensor([0, 1, 1, 1])
        >>> average_precision = BinnedAveragePrecision(num_classes=1, num_thresholds=10)
        >>> average_precision(pred, target)
        tensor(1.0000)

    Example (multiclass case):
        >>> pred = torch.tensor([[0.75, 0.05, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.75, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.75, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.05, 0.75, 0.05]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> average_precision = BinnedAveragePrecision(num_classes=5, num_thresholds=10)
        >>> average_precision(pred, target)
        [tensor(1.0000), tensor(1.0000), tensor(0.2500), tensor(0.2500), tensor(-0.)]
    """

    def compute(self) -> Union[List[Tensor], Tensor]:
        precisions, recalls, _ = super().compute()
        return _average_precision_compute_with_precision_recall(precisions, recalls, self.num_classes)


class BinnedRecallAtFixedPrecision(BinnedPrecisionRecallCurve):
    """
    Computes the higest possible recall value given the minimum precision thresholds provided.

    Computation is performed in constant-memory by computing precision and recall
    for ``num_thresholds`` buckets/thresholds (evenly distributed between 0 and 1).

    Forward accepts

    - ``preds`` (float tensor): ``(N, ...)`` (binary) or ``(N, C, ...)`` (multiclass) tensor
      with probabilities, where C is the number of classes.

    - ``target`` (long tensor): ``(N, ...)`` with integer labels

    Args:
        num_classes: integer with number of classes. Provide 1 for for binary problems.
        min_precision: float value specifying minimum precision threshold.
        num_thresholds: number of bins used for computation. More bins will lead to more detailed
            curve and accurate estimates, but will be slower and consume more memory. Default 100
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)

    Example (binary case):
        >>> from torchmetrics import BinnedRecallAtFixedPrecision
        >>> pred = torch.tensor([0, 0.2, 0.5, 0.8])
        >>> target = torch.tensor([0, 1, 1, 0])
        >>> average_precision = BinnedRecallAtFixedPrecision(num_classes=1, num_thresholds=10, min_precision=0.5)
        >>> average_precision(pred, target)
        (tensor(1.0000), tensor(0.1111))

    Example (multiclass case):
        >>> pred = torch.tensor([[0.75, 0.05, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.75, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.75, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.05, 0.75, 0.05]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> average_precision = BinnedRecallAtFixedPrecision(num_classes=5, num_thresholds=10, min_precision=0.5)
        >>> average_precision(pred, target)   # doctest: +NORMALIZE_WHITESPACE
        (tensor([1.0000, 1.0000, 0.0000, 0.0000, 0.0000]),
        tensor([6.6667e-01, 6.6667e-01, 1.0000e+06, 1.0000e+06, 1.0000e+06]))
    """

    def __init__(
        self,
        num_classes: int,
        min_precision: float,
        num_thresholds: int = 100,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        super().__init__(
            num_classes=num_classes,
            num_thresholds=num_thresholds,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )
        self.min_precision = min_precision

    def compute(self) -> Tuple[Tensor, Tensor]:
        """Returns float tensor of size n_classes"""
        precisions, recalls, thresholds = super().compute()

        if self.num_classes == 1:
            return _recall_at_precision(precisions, recalls, thresholds, self.min_precision)

        recalls_at_p = torch.zeros(self.num_classes, device=recalls[0].device, dtype=recalls[0].dtype)
        thresholds_at_p = torch.zeros(self.num_classes, device=thresholds[0].device, dtype=thresholds[0].dtype)
        for i in range(self.num_classes):
            recalls_at_p[i], thresholds_at_p[i] = _recall_at_precision(
                precisions[i], recalls[i], thresholds[i], self.min_precision
            )
        return (recalls_at_p, thresholds_at_p)
