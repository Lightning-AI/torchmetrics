# Copyright The Lightning team.
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
from typing import Literal

import torch
from torch import Tensor, tensor

from torchmetrics.functional.clustering.mutual_info_score import _mutual_info_score_compute, _mutual_info_score_update
from torchmetrics.functional.clustering.utils import (
    _validate_average_method_arg,
    calculate_entropy,
    calculate_generalized_mean,
)


def adjusted_mutual_info_score(
    preds: Tensor, target: Tensor, average_method: Literal["min", "geometric", "arithmetic", "max"] = "arithmetic"
) -> Tensor:
    """Compute adjusted mutual information between two clusterings.

    Args:
        preds: predicted cluster labels
        target: ground truth cluster labels
        average_method: normalizer computation method

    Returns:
        Scalar tensor with adjusted mutual info score between 0.0 and 1.0

    Example:
        >>> from torchmetrics.functional.clustering import adjusted_mutual_info_score
        >>> preds = torch.tensor([2, 1, 0, 1, 0])
        >>> target = torch.tensor([0, 2, 1, 1, 0])
        >>> adjusted_mutual_info_score(preds, target, "arithmetic")
        tensor(-0.2500)

    """
    _validate_average_method_arg(average_method)
    contingency = _mutual_info_score_update(preds, target)
    mutual_info = _mutual_info_score_compute(contingency)
    expected_mutual_info = expected_mutual_info_score(contingency, target.numel())
    normalizer = calculate_generalized_mean(
        torch.stack([calculate_entropy(preds), calculate_entropy(target)]), average_method
    )
    denominator = normalizer - expected_mutual_info
    if denominator < 0:
        denominator = torch.min(torch.tensor([denominator, -torch.finfo(denominator.dtype).eps]))
    else:
        denominator = torch.max(torch.tensor([denominator, torch.finfo(denominator.dtype).eps]))

    return (mutual_info - expected_mutual_info) / denominator


def expected_mutual_info_score(contingency: Tensor, n_samples: int) -> Tensor:
    """Calculated expected mutual information score between two clusterings.

    Implementation taken from sklearn/metrics/cluster/_expected_mutual_info_fast.pyx.

    Args:
        contingency: contingency matrix
        n_samples: number of samples

    Returns:
        expected_mutual_info_score: expected mutual information score

    """
    n_rows, n_cols = contingency.shape
    a = torch.ravel(contingency.sum(dim=1))
    b = torch.ravel(contingency.sum(dim=0))

    # Check if preds or target labels only have one cluster
    if a.numel() == 1 or b.numel() == 1:
        return tensor(0.0, device=a.device)

    nijs = torch.arange(0, max([a.max().item(), b.max().item()]) + 1, device=a.device)
    nijs[0] = 1

    term1 = nijs / n_samples
    log_a = torch.log(a)
    log_b = torch.log(b)

    log_nnij = torch.log(torch.tensor(n_samples, device=a.device)) + torch.log(nijs)

    gln_a = torch.lgamma(a + 1)
    gln_b = torch.lgamma(b + 1)
    gln_na = torch.lgamma(n_samples - a + 1)
    gln_nb = torch.lgamma(n_samples - b + 1)
    gln_nnij = torch.lgamma(nijs + 1) + torch.lgamma(torch.tensor(n_samples + 1, dtype=a.dtype, device=a.device))

    emi = tensor(0.0, device=a.device)
    for i in range(n_rows):
        for j in range(n_cols):
            start = int(max(1, a[i].item() - n_samples + b[j].item()))
            end = int(min(a[i].item(), b[j].item()) + 1)

            for nij in range(start, end):
                term2 = log_nnij[nij] - log_a[i] - log_b[j]
                gln = (
                    gln_a[i]
                    + gln_b[j]
                    + gln_na[i]
                    + gln_nb[j]
                    - gln_nnij[nij]
                    - torch.lgamma(a[i] - nij + 1)
                    - torch.lgamma(b[j] - nij + 1)
                    - torch.lgamma(n_samples - a[i] - b[j] + nij + 1)
                )
                term3 = torch.exp(gln)
                emi += term1[nij] * term2 * term3

    return emi
