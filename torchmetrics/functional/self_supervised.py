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
from warnings import warn

import torch
from torch import Tensor

from torchmetrics.functional.pairwise import pairwise_cosine_similarity, pairwise_linear_similarity


def embedding_similarity(
    batch: Tensor, similarity: str = "cosine", reduction: str = "none", zero_diagonal: bool = True
) -> Tensor:
    """Computes representation similarity.

    Example:
        >>> from torchmetrics.functional import embedding_similarity
        >>> embeddings = torch.tensor([[1., 2., 3., 4.], [1., 2., 3., 4.], [4., 5., 6., 7.]])
        >>> embedding_similarity(embeddings)
        tensor([[0.0000, 1.0000, 0.9759],
                [1.0000, 0.0000, 0.9759],
                [0.9759, 0.9759, 0.0000]])

    Args:
        batch: (batch, dim)
        similarity: 'dot' or 'cosine'
        reduction: 'none', 'sum', 'mean' (all along dim -1)
        zero_diagonal: if True, the diagonals are set to zero

    Return:
        A square matrix (batch, batch) with the similarity scores between all elements
        If sum or mean are used, then returns (b, 1) with the reduced value for each row

    .. deprecated:: v0.6
        Use :func:`torchmetrics.functional.pairwise_cosine_similarity` when `similarity='cosine'`
        else use :func:`torchmetrics.functional.pairwise_euclidean_distance`. Will be removed in v0.7.
    """
    warn(
        "Function `embedding_similarity` was deprecated v0.6 and will be removed in v0.7."
        " Use `torchmetrics.functional.pairwise_cosine_similarity` instead when argument"
        " similarity='cosine' else use `torchmetrics.functional.pairwise_linear_similarity",
        DeprecationWarning,
    )
    if similarity == "cosine":
        return pairwise_cosine_similarity(batch, reduction=reduction, zero_diagonal=zero_diagonal)
    return pairwise_linear_similarity(batch, reduction=reduction, zero_diagonal=zero_diagonal)
