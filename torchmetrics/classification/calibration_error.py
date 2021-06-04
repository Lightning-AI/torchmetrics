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
from typing import Optional, Callable, Any

import torch
from torch import Tensor
from torchmetrics.metric import Metric
from torchmetrics.functional.classification.calibration_error import _ce_compute, _ce_update
from torchmetrics.utilities import rank_zero_warn


class ExpectedCalibrationError(Metric):
    def __init__(self, n_bins: int = 15, compute_on_step: bool = False, dist_sync_on_step: bool = False, process_group: Optional[Any] = None, dist_sync_fn: Callable = None):
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
                         process_group=process_group, dist_sync_fn=dist_sync_fn)

        self.n_bins = n_bins
        self.bin_boundaries = torch.linspace(0, 1, n_bins + 1)

        self.add_state("confidences", list(), dist_reduce_fx=None)
        self.add_state("accuracies", list(), dist_reduce_fx=None)

        # TODO: rank zero warning?

    def update(self, preds: Tensor, target: Tensor):
        """[summary]

        Args:
            preds (Tensor): [description]
            target (Tensor): [description]
        """
        confidences, accuracies = _ce_update(preds, target)

        self.confidences.append(confidences)
        self.accuracies.append(accuracies)

    def compute(self) -> Tensor:
        """[summary]

        Returns:
            Tensor: [description]
        """
        confidences = torch.cat(self.confidences, dim=0)
        accuracies = torch.cat(self.accuracies, dim=0)
        return _ce_compute(confidences, accuracies, self.bin_boundaries)


class MaximumCalibrationError(Metric):
    def __init__(self, n_bins: int = 15, compute_on_step: bool = False, dist_sync_on_step: bool = False, process_group: Optional[Any] = None, dist_sync_fn: Callable = None):
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
                         process_group=process_group, dist_sync_fn=dist_sync_fn)

        self.n_bins = n_bins
        self.bin_boundaries = torch.linspace(0, 1, n_bins + 1)

        self.add_state("confidences", list(), dist_reduce_fx=None)
        self.add_state("accuracies", list(), dist_reduce_fx=None)

        # TODO: rank zero warning?

    def update(self, preds: Tensor, target: Tensor):
        """[summary]

        Args:
            preds (Tensor): [description]
            target (Tensor): [description]
        """
        confidences, accuracies = _ce_update(preds, target)

        self.confidences.append(confidences)
        self.accuracies.append(accuracies)

    def compute(self) -> Tensor:
        """[summary]

        Returns:
            Tensor: [description]
        """
        confidences = torch.cat(self.confidences, dim=0)
        accuracies = torch.cat(self.accuracies, dim=0)
        return _ce_compute(confidences, accuracies, self.bin_boundaries, norm="max")
