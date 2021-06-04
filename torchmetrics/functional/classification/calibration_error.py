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
from typing import Optional, Tuple

import torch
from torch import Tensor, tensor
from torch.nn import functional as F

from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.checks import _input_format_classification
from torchmetrics.utilities.enums import DataType


def _ce_compute(confidences: Tensor, accuracies: Tensor, bin_boundaries: Tensor, norm: str = "l1") -> Tensor:

    conf_bin = torch.zeros_like(bin_boundaries)
    acc_bin = torch.zeros_like(bin_boundaries)
    prop_bin = torch.zeros_like(bin_boundaries)
    for i, (bin_lower, bin_upper) in enumerate(zip(bin_boundaries[:-1], bin_boundaries[1:])):
        # Calculated confidence and accuracy in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            acc_bin[i] = accuracies[in_bin].float().mean()
            conf_bin[i] = confidences[in_bin].mean()
            prop_bin[i] = prop_in_bin

    if norm == "l1":
        ce = torch.sum(torch.abs(conf_bin - acc_bin) * prop_bin)
    elif norm == "max":
        ce = torch.max(torch.abs(conf_bin - acc_bin))
    else:
        raise ValueError(f"Norm {norm} is not supported.")

    return ce


def _ce_update(preds: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    _, _, mode = _input_format_classification(preds, target)

    if mode == DataType.BINARY:
        confidences, accuracies = preds, target
    elif mode == DataType.MULTICLASS:
        confidences, predictions = preds.max(dim=1)
        accuracies = predictions.eq(target)
    elif mode == DataType.MULTIDIM_MULTICLASS:
        # reshape tensors
        # for preds, move the class dimension to the final axis and flatten the rest
        confidences, predictions = torch.transpose(preds, 1, -1).flatten(0, -2).max(dim=1)
        # for targets, just flatten the target
        accuracies = predictions.eq(target.flatten())
    else:
        raise ValueError(
            f"Calibration error is not well-defined for data with size {preds.size()} and targets {target.size()}")

    return confidences.float(), accuracies.float()


def maximum_calibration_error(preds: Tensor, target: Tensor, n_bins: int = 15) -> Tensor:
    """

    TODO: docstring

    Args:
        preds (Tensor): Input probabilities.
        target (Tensor): Class labels.
        n_bins (int, optional): [description]. Defaults to 15.
    """

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)

    confidences, accuracies = _ce_update(preds, target)

    return _ce_compute(confidences, accuracies, bin_boundaries, norm="max")


def expected_calibration_error(preds: Tensor, target: Tensor, n_bins: int = 15) -> Tensor:
    """

    TODO: docstring

    Args:
        preds (Tensor): Input probabilities.
        target (Tensor): Class labels.
        n_bins (int, optional): [description]. Defaults to 15.
    """

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)

    confidences, accuracies = _ce_update(preds, target)
    # raise ValueError
    return _ce_compute(confidences, accuracies, bin_boundaries)
