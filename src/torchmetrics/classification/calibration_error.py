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
from typing import Any, List

import torch
from torch import Tensor

from torchmetrics.functional.classification.calibration_error import _ce_compute, _ce_update
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat


class CalibrationError(Metric):
    r"""`Computes the Top-label Calibration Error`_
    Three different norms are implemented, each corresponding to variations on the calibration error metric.

    L1 norm (Expected Calibration Error)

    .. math::
        \text{ECE} = \sum_i^N b_i \|(p_i - c_i)\|

    Infinity norm (Maximum Calibration Error)

    .. math::
        \text{MCE} =  \max_{i} (p_i - c_i)

    L2 norm (Root Mean Square Calibration Error)

    .. math::
        \text{RMSCE} = \sqrt{\sum_i^N b_i(p_i - c_i)^2}

    Where :math:`p_i` is the top-1 prediction accuracy in bin :math:`i`,
    :math:`c_i` is the average confidence of predictions in bin :math:`i`, and
    :math:`b_i` is the fraction of data points in bin :math:`i`.

    .. note::
        L2-norm debiasing is not yet supported.

    Args:
        n_bins: Number of bins to use when computing probabilities and accuracies.
        norm: Norm used to compare empirical and expected probability bins.
            Defaults to "l1", or Expected Calibration Error.
        debias: Applies debiasing term, only implemented for l2 norm. Defaults to True.

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.
    """
    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False
    DISTANCES = {"l1", "l2", "max"}
    confidences: List[Tensor]
    accuracies: List[Tensor]

    def __init__(
        self,
        n_bins: int = 15,
        norm: str = "l1",
        **kwargs: Any,
    ):

        super().__init__(**kwargs)

        if norm not in self.DISTANCES:
            raise ValueError(f"Norm {norm} is not supported. Please select from l1, l2, or max. ")

        if not isinstance(n_bins, int) or n_bins <= 0:
            raise ValueError(f"Expected argument `n_bins` to be a int larger than 0 but got {n_bins}")
        self.n_bins = n_bins
        self.bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.norm = norm

        self.add_state("confidences", [], dist_reduce_fx="cat")
        self.add_state("accuracies", [], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Computes top-level confidences and accuracies for the input probabilities and appends them to internal
        state.

        Args:
            preds (Tensor): Model output probabilities.
            target (Tensor): Ground-truth target class labels.
        """
        confidences, accuracies = _ce_update(preds, target)

        self.confidences.append(confidences)
        self.accuracies.append(accuracies)

    def compute(self) -> Tensor:
        """Computes calibration error across all confidences and accuracies.

        Returns:
            Tensor: Calibration error across previously collected examples.
        """
        confidences = dim_zero_cat(self.confidences)
        accuracies = dim_zero_cat(self.accuracies)
        return _ce_compute(confidences, accuracies, self.bin_boundaries.to(self.device), norm=self.norm)
