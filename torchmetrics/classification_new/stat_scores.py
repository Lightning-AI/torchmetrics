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
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from torchmetrics.functional.classification_new.stat_scores import (
    _binary_stat_scores_arg_validation,
    _binary_stat_scores_compute,
    _binary_stat_scores_format,
    _binary_stat_scores_tensor_validation,
    _binary_stat_scores_update,
)
from torchmetrics.metric import Metric


class BinaryStatScores(Metric):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        threshold: float = 0.5,
        multidim_average: str = "global",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _binary_stat_scores_arg_validation(threshold, multidim_average, ignore_index)
        self.threshold = threshold
        self.multidim_average = multidim_average
        self.ignore_index = ignore_index
        self.validate_args = validate_args

        if self.multidim_average == "samplewise":
            self.add_state("tp", [], dist_reduce_fx="cat")
            self.add_state("fp", [], dist_reduce_fx="cat")
            self.add_state("tn", [], dist_reduce_fx="cat")
            self.add_state("fn", [], dist_reduce_fx="cat")
        else:
            self.add_state("tp", torch.zeros(1), dist_reduce_fx="sum")
            self.add_state("fp", torch.zeros(1), dist_reduce_fx="sum")
            self.add_state("tn", torch.zeros(1), dist_reduce_fx="sum")
            self.add_state("fn", torch.zeros(1), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        if self.validate_args:
            _binary_stat_scores_tensor_validation(preds, target, self.multidim_average, self.ignore_index)
        preds, target = _binary_stat_scores_format(preds, target, self.threshold, self.ignore_index)
        tp, fp, tn, fn = _binary_stat_scores_update(preds, target, self.multidim_average)
        if self.multidim_average == "samplewise":
            self.tp.append(tp)
            self.fp.append(fp)
            self.tn.append(tn)
            self.fn.append(fn)
        else:
            self.tp += tp
            self.fp += fp
            self.tn += tn
            self.fn += fn

    def compute(self) -> Tensor:
        return _binary_stat_scores_compute(self.tp, self.fp, self.tn, self.fn, self.multidim_average)
