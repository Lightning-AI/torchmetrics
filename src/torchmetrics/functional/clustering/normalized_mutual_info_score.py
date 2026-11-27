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
from torch import Tensor

from torchmetrics.functional.clustering.mutual_info_score import _mutual_info_score_compute, _mutual_info_score_update
from torchmetrics.functional.clustering.utils import (
    _validate_average_method_arg,
    calculate_entropy,
    calculate_generalized_mean,
)


def normalized_mutual_info_score(
    preds: Tensor, target: Tensor, average_method: Literal["min", "geometric", "arithmetic", "max"] = "arithmetic"
) -> Tensor:
    """Compute normalized mutual information between two clusterings.

    Args:
        preds: predicted cluster labels
        target: ground truth cluster labels
        average_method: normalizer computation method

    Returns:
        Scalar tensor with normalized mutual info score between 0.0 and 1.0

    Example:
        >>> from torchmetrics.functional.clustering import normalized_mutual_info_score
        >>> target = torch.tensor([0, 3, 2, 2, 1])
        >>> preds = torch.tensor([1, 3, 2, 0, 1])
        >>> normalized_mutual_info_score(preds, target, "arithmetic")
        tensor(0.7919)

    """
    _validate_average_method_arg(average_method)
    contingency = _mutual_info_score_update(preds, target)
    if contingency.numel() == 0 or contingency.shape == (1, 1):
        return torch.tensor(1.0, device=preds.device)
    mutual_info = _mutual_info_score_compute(contingency)
    if torch.allclose(mutual_info, torch.zeros_like(mutual_info), atol=torch.finfo(mutual_info.dtype).eps):
        return mutual_info

    normalizer = calculate_generalized_mean(
        torch.stack([calculate_entropy(preds), calculate_entropy(target)]), average_method
    )

    return mutual_info / normalizer
