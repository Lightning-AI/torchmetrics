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
from typing import Optional

import torch
from torch import Tensor, tensor

from torchmetrics.utilities.checks import _check_retrieval_functional_inputs


def retrieval_reciprocal_rank(preds: Tensor, target: Tensor, top_k: Optional[int] = None) -> Tensor:
    """Compute reciprocal rank (for information retrieval). See `Mean Reciprocal Rank`_.

    ``preds`` and ``target`` should be of the same shape and live on the same device. If no ``target`` is ``True``,
    0 is returned. ``target`` must be either `bool` or `integers` and ``preds`` must be ``float``,
    otherwise an error is raised.

    Args:
        preds: estimated probabilities of each document to be relevant.
        target: ground truth about each document being relevant or not.
        top_k: consider only the top k elements (default: ``None``, which considers them all)

    Return:
        a single-value tensor with the reciprocal rank (RR) of the predictions ``preds`` wrt the labels ``target``.

    Raises:
        ValueError:
            If ``top_k`` is not ``None`` or an integer larger than 0.

    Example:
        >>> from torchmetrics.functional.retrieval import retrieval_reciprocal_rank
        >>> preds = torch.tensor([0.2, 0.3, 0.5])
        >>> target = torch.tensor([False, True, False])
        >>> retrieval_reciprocal_rank(preds, target)
        tensor(0.5000)

    """
    preds, target = _check_retrieval_functional_inputs(preds, target)

    top_k = top_k or preds.shape[-1]
    if not isinstance(top_k, int) and top_k <= 0:
        raise ValueError(f"Argument ``top_k`` has to be a positive integer or None, but got {top_k}.")

    target = target[preds.topk(min(top_k, preds.shape[-1]), sorted=True, dim=-1)[1]]
    if not target.sum():
        return tensor(0.0, device=preds.device)

    position = torch.nonzero(target).view(-1)
    return 1.0 / (position[0] + 1.0)
