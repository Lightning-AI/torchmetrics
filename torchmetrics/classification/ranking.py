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
from typing import Optional

import torch
from torch import Tensor

from torchmetrics.functional.classification.ranking import (
    _coverage_error_compute,
    _coverage_error_update,
    _label_ranking_average_precision_compute,
    _label_ranking_average_precision_update,
    _label_ranking_loss_compute,
    _label_ranking_loss_update,
)
from torchmetrics.metric import Metric


class CoverageError(Metric):

    higher_is_better: bool = False
    is_differentiable: bool = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("coverage", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("numel", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("weight", torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, sample_weight: Optional[Tensor] = None) -> None:
        coverage, numel, sample_weight = _coverage_error_update(preds, target, sample_weight)
        self.coverage += coverage
        self.numel += numel
        if sample_weight is not None:
            self.weight += sample_weight

    def compute(self) -> Tensor:
        return _coverage_error_compute(self.coverage, self.numel, self.weight)


class LabelRankingAveragePrecisionScore(Metric):

    higher_is_better: bool = False
    is_differentiable: bool = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("score", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("numel", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sample_weight", torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, sample_weight: Optional[Tensor] = None) -> None:
        score, numel, sample_weight = _label_ranking_average_precision_update(preds, target, sample_weight)
        self.score += score
        self.numel += numel
        if sample_weight is not None:
            self.sample_weight += sample_weight

    def compute(self) -> Tensor:
        return _label_ranking_average_precision_compute(self.score, self.numel, self.sample_weight)


class LabelRankingLoss(Metric):
    higher_is_better: bool = False
    is_differentiable: bool = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("loss", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("numel", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sample_weight", torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, sample_weight: Optional[Tensor] = None) -> None:
        loss, numel, sample_weight = _label_ranking_loss_update(preds, target, sample_weight)
        self.loss += loss
        self.numel += numel
        if sample_weight is not None:
            self.sample_weight += sample_weight

    def compute(self) -> Tensor:
        return _label_ranking_loss_compute(self.loss, self.numel, self.sample_weight)
