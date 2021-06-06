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


class CalibrationError(Metric):

    def __init__(self, n_bins: int = 15, norm: str = "l1", debias: bool = True, compute_on_step: bool = False, dist_sync_on_step: bool = False, process_group: Optional[Any] = None, dist_sync_fn: Callable = None):
        """

        Computes the top-label calibration error as described in `https://arxiv.org/pdf/1909.10155.pdf`. 

        Three different norms are implemented, each corresponding to variations on the calibration error metric.

        L1 norm (Expected Calibration Error)

        .. math::
            \text{Accuracy} = \frac{1}{N}\sum_i^N 1(y_i = \hat{y}_i)


        Infinity norm (Maximum Calibration Error)

        .. math::
        \text{Accuracy} = \frac{1}{N}\sum_i^N 1(y_i = \hat{y}_i)

        L2 norm (Root Mean Square Calibration Error)

        .. math::
        \text{Accuracy} = \frac{1}{N}\sum_i^N 1(y_i = \hat{y}_i)

        Debiasing is only supported for the L2 norm, and adds an additional term to the calibration error:

        .. math::
        \text{Accuracy} = \frac{1}{N}\sum_i^N 1(y_i = \hat{y}_i)




        Args:
            n_bins (int, optional): Number of bins to use when computing t. Defaults to 15.
            norm (str, optional): Norm used to compare empirical and expected probability bins. 
                Defaults to "l1", or Expected Calibration Error.
            debias (bool, optional): Applies debiasing term, only implemented for l2 norm. Defaults to True.
            compute_on_step (bool, optional):  Forward only calls ``update()`` and return None if this is set to False. Defaults to False.
            dist_sync_on_step (bool, optional): Synchronize metric state across processes at each ``forward()``
                before returning the value at the step.. Defaults to False.
            process_group (Optional[Any], optional): Specify the process group on which synchronization is called. default: None (which selects the entire world). Defaults to None.
            dist_sync_fn (Callable, optional): Callback that performs the ``allgather`` operation on the metric state. When ``None``, DDP
                will be used to perform the ``allgather``.. Defaults to None.
        """
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
                         process_group=process_group, dist_sync_fn=dist_sync_fn)

        if norm not in ["l1", "l2", "max"]:
            raise ValueError(f"Norm {norm} is not supported. Please select from l1, l2, or max. ")

        self.n_bins = n_bins
        self.bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.norm = norm
        self.debias = debias

        self.add_state("confidences", list(), dist_reduce_fx=None)
        self.add_state("accuracies", list(), dist_reduce_fx=None)

    def update(self, preds: Tensor, target: Tensor):
        """
        Computes top-level confidences and accuracies for the input probabilites and appends them to internal state.

        Args:
            preds (Tensor): [description]
            target (Tensor): [description]
        """
        confidences, accuracies = _ce_update(preds, target)

        self.confidences.append(confidences)
        self.accuracies.append(accuracies)

    def compute(self) -> Tensor:
        """
        Computes calibration error across all confidences and accuracies.

        Returns:
            Tensor: [description]
        """
        confidences = torch.cat(self.confidences, dim=0)
        accuracies = torch.cat(self.accuracies, dim=0)
        return _ce_compute(confidences, accuracies, self.bin_boundaries, norm=self.norm, debias=self.debias)
