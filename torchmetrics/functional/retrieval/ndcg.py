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
import torch
from torch import Tensor, tensor

from torchmetrics.utilities.checks import _check_retrieval_functional_inputs


def _dcg(target):
    denom = torch.log2(torch.arange(target.shape[-1], device=target.device) + 2.0)
    return (target / denom).sum()


def retrieval_normalized_dcg(preds: Tensor, target: Tensor, k: int = None) -> Tensor:
    """
    Computes Normalized Discounted Cumulative Gain (for information retrieval), as explained
    `here <https://en.wikipedia.org/wiki/Discounted_cumulative_gain>`__.

    ``preds`` and ``target`` should be of the same shape and live on the same device.
    ``target`` must be either `bool` or `integers` and ``preds`` must be `float`,
    otherwise an error is raised.

    Args:
        preds: estimated probabilities of each document to be relevant.
        target: ground truth about each document relevance.
        k: consider only the top k elements (default: None)

    Return:
        a single-value tensor with the nDCG of the predictions ``preds`` w.r.t. the labels ``target``.

    Example:
        >>> from torchmetrics.functional import retrieval_normalized_dcg
        >>> preds = torch.tensor([.1, .2, .3, 4, 70])
        >>> target = torch.tensor([10, 0, 0, 1, 5])
        >>> retrieval_normalized_dcg(preds, target)
        tensor(0.6957)
    """
    preds, target = _check_retrieval_functional_inputs(preds, target, allow_non_binary_target=True)

    k = preds.shape[-1] if k is None else k

    if not (isinstance(k, int) and k > 0):
        raise ValueError("`k` has to be a positive integer or None")

    if not target.sum():
        return tensor(0.0, device=preds.device)

    sorted_target = target[torch.argsort(preds, dim=-1, descending=True)][:k]
    ideal_target = torch.sort(target, descending=True)[0][:k]

    return _dcg(sorted_target) / _dcg(ideal_target)
