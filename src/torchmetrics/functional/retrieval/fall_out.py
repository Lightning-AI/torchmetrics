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


def retrieval_fall_out(preds: Tensor, target: Tensor, top_k: Optional[int] = None) -> Tensor:
    """Compute the Fall-out for information retrieval, as explained in `IR Fall-out`_.

    Fall-out is the fraction of non-relevant documents retrieved among all the non-relevant documents.

    ``preds`` and ``target`` should be of the same shape and live on the same device. If no ``target`` is ``True``,
    ``0`` is returned. ``target`` must be either `bool` or `integers` and ``preds`` must be ``float``,
    otherwise an error is raised. If you want to measure Fall-out@K, ``top_k`` must be a positive integer.

    Args:
        preds: estimated probabilities of each document to be relevant.
        target: ground truth about each document being relevant or not.
        top_k: consider only the top k elements (default: ``None``, which considers them all)

    Returns:
        A single-value tensor with the fall-out (at ``top_k``) of the predictions ``preds`` w.r.t. the labels ``target``

    Raises:
        ValueError:
            If ``top_k`` parameter is not `None` or an integer larger than 0

    Example:
        >>> from  torchmetrics.functional import retrieval_fall_out
        >>> preds = tensor([0.2, 0.3, 0.5])
        >>> target = tensor([True, False, True])
        >>> retrieval_fall_out(preds, target, top_k=2)
        tensor(1.)
    """
    preds, target = _check_retrieval_functional_inputs(preds, target)

    top_k = preds.shape[-1] if top_k is None else top_k

    if not (isinstance(top_k, int) and top_k > 0):
        raise ValueError("`top_k` has to be a positive integer or None")

    target = 1 - target  # we want to compute the probability of getting a non-relevant doc among all non-relevant docs

    if not target.sum():
        return tensor(0.0, device=preds.device)

    relevant = target[torch.argsort(preds, dim=-1, descending=True)][:top_k].sum().float()
    return relevant / target.sum()
