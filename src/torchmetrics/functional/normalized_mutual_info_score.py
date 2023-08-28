
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
import math
from typing import Literal, Tuple

import torch
from torch import Tensor

from torchmetrics.functional.clustering.mutual_info_score import _mutual_info_score_update, _mutual_info_score_compute


def normalized_mutual_info_score(
    preds: Tensor,
    target: Tensor,
    method: Literal["min", "geometric", "arithmetic", "max"] = "arithmetic"
) -> Tensor:
    """Compute normalized mutual information between two clusterings.

    Args:
        preds: predicted classes
        target: ground truth classes
        method: normalizer computation method

    Returns:
        normalized_mutual_info_score: score between 0.0 and 1.0

    Example:
        >>> from torchmetrics.functional.clustering import normalized_mutual_info_score
        >>> target = torch.tensor([0, 3, 2, 2, 1])
        >>> preds = torch.tensor([1, 3, 2, 0, 1])
        >>> normalized_mutual_info_score(preds, target, "arithmetic")
        tensor(0.5)

    """
    contingency = _mutual_info_score_update(preds, target)
    normalizer = generalized_average(entropy(preds), entropy(target), method)
    return _mutual_info_score_compute(contingency) / normalizer


def entropy(labels: Tensor) -> Tensor:
    return tensor(0.)


def generalized_average(
    preds: Tensor,
    target: Tensor,
    method:Literal["min", "geometric", "arithmetic", "max"]
) -> Tensor:
    return tensor(0.)
