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
from typing import Tuple

import torch
from torch import Tensor

import torchmetrics
from torchmetrics.utilities.checks import _check_same_shape


def _cosine_similarity_update(
    preds: Tensor,
    target: Tensor,
) -> Tuple[Tensor, int]:
    preds.float()
    target.float()

    return preds, target


def _cosine_similarity_compute(preds: Tensor, target: Tensor, reduction='sum') -> Tensor:
    dot_product = (preds * target).sum(dim=-1)
    preds_norm = preds.norm(dim=-1)
    target_norm = target.norm(dim=-1)
    similarity = dot_product / (preds_norm * target_norm)
    reduction_mapping = {"sum": torch.sum, "mean": torch.mean, "none": lambda x: x}
    return reduction_mapping[reduction](similarity)


def cosine_similarity(preds: Tensor, target: Tensor, reduction='sum') -> Tensor:
    r"""
        Computes the `Cosine Similarity <https://en.wikipedia.org/wiki/Cosine_similarity>`_
        between targets and predictions:
        Accepts all input types listed in :ref:`references/modules:input types`.
        Args:
            preds: Predictions from model (probabilities, logits or labels)
            target: Ground truth
            reduction: The method of reducing along the batch dimension using sum, mean or
                        taking the individual scores

        Example:
            >>> from torchmetrics.functional import cosine_similarity
            >>> target = torch.tensor([1, 2, 3, 4], dtype= torch.float)
            >>> preds = torch.tensor([[1, 2, 3, 4],
            ...                       [1, 2, 3, 4],
            ...                       [1, 2, 3, 4]],
            ...                       dtype= torch.float)
            >>> cosine_similarity(preds, target, 'none')
            tensor([1.0000, 1.0000, 1.0000])

        """
    preds, target = _cosine_similarity_update(preds, target)
    return _cosine_similarity_compute(preds, target, reduction)
