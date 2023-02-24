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
import torch
from torch import Tensor, tensor

from torchmetrics.utilities.checks import _check_retrieval_functional_inputs


def retrieval_r_precision(preds: Tensor, target: Tensor) -> Tensor:
    """Compute the r-precision metric for information retrieval.

    R-Precision is the fraction of relevant documents among all the top ``k`` retrieved documents where ``k`` is equal
    to the total number of relevant documents.

    ``preds`` and ``target`` should be of the same shape and live on the same device. If no ``target`` is ``True``,
    ``0`` is returned. ``target`` must be either `bool` or `integers` and ``preds`` must be ``float``,
    otherwise an error is raised. If you want to measure Precision@K, ``top_k`` must be a positive integer.

    Args:
        preds: estimated probabilities of each document to be relevant.
        target: ground truth about each document being relevant or not.

    Returns:
        A single-value tensor with the r-precision of the predictions ``preds`` w.r.t. the labels ``target``.

    Example:
        >>> preds = tensor([0.2, 0.3, 0.5])
        >>> target = tensor([True, False, True])
        >>> retrieval_r_precision(preds, target)
        tensor(0.5000)
    """
    preds, target = _check_retrieval_functional_inputs(preds, target)

    relevant_number = target.sum()
    if not relevant_number:
        return tensor(0.0, device=preds.device)

    relevant = target[torch.argsort(preds, dim=-1, descending=True)][:relevant_number].sum().float()
    return relevant / relevant_number
