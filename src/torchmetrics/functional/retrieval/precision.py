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

from torch import Tensor, tensor

from torchmetrics.utilities.checks import _check_retrieval_functional_inputs


def retrieval_precision(preds: Tensor, target: Tensor, top_k: Optional[int] = None, adaptive_k: bool = False) -> Tensor:
    """Compute the precision metric for information retrieval.

    Precision is the fraction of relevant documents among all the retrieved documents.

    ``preds`` and ``target`` should be of the same shape and live on the same device. If no ``target`` is ``True``,
    ``0`` is returned. ``target`` must be either `bool` or `integers` and ``preds`` must be ``float``,
    otherwise an error is raised. If you want to measure Precision@K, ``top_k`` must be a positive integer.

    Args:
        preds: estimated probabilities of each document to be relevant.
        target: ground truth about each document being relevant or not.
        top_k: consider only the top k elements (default: ``None``, which considers them all)
        adaptive_k: adjust `k` to `min(k, number of documents)` for each query

    Returns:
        A single-value tensor with the precision (at ``top_k``) of the predictions ``preds`` w.r.t. the labels
          ``target``.

    Raises:
        ValueError:
            If ``top_k`` is not `None` or an integer larger than 0.
        ValueError:
            If ``adaptive_k`` is not boolean.

    Example:
        >>> preds = tensor([0.2, 0.3, 0.5])
        >>> target = tensor([True, False, True])
        >>> retrieval_precision(preds, target, top_k=2)
        tensor(0.5000)
    """
    preds, target = _check_retrieval_functional_inputs(preds, target)

    if not isinstance(adaptive_k, bool):
        raise ValueError("`adaptive_k` has to be a boolean")

    if top_k is None or (adaptive_k and top_k > preds.shape[-1]):
        top_k = preds.shape[-1]

    if not (isinstance(top_k, int) and top_k > 0):
        raise ValueError("`top_k` has to be a positive integer or None")

    if not target.sum():
        return tensor(0.0, device=preds.device)

    relevant = target[preds.topk(min(top_k, preds.shape[-1]), dim=-1)[1]].sum().float()
    return relevant / top_k
