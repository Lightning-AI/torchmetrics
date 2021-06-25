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
from typing import Any, Callable, Optional

import torch
from torch import Tensor, tensor

from torchmetrics.functional.classification.cosine_similarity import (
    _cosine_similarity_compute,
    _cosine_similarity_update,
)
from torchmetrics.metric import Metric


class CosineSimilarity(Metric):
    r"""
       Computes the `Cosine Similarity <https://en.wikipedia.org/wiki/Cosine_similarity>`_
        between targets and predictions:
        
        .. math::
            cos_{sim}(x,y) = \frac{x \cdot y}{||x|| \cdot ||y|| = \frac{\sum_{i=1}^n x_i y_i}{\sqrt{\sum_{i=1}^n x_i^2} \sqrt{\sum_{i=1}^n y_i^2}}
            
        where :math:`y` is a tensor of target values, and :math:`x` is a tensor of predictions.
       Accepts all input types listed in :ref:`references/modules:input types`.

       Args:
           reduction : how to reduce over the batch dimension using sum, mean or
                        taking the individual scores
           compute_on_step:
               Forward only calls ``update()`` and return ``None`` if this is set to ``False``.
           dist_sync_on_step:
               Synchronize metric state across processes at each ``forward()``
               before returning the value at the step.
           process_group:
               Specify the process group on which synchronization is called.
               default: ``None`` (which selects the entire world)
           dist_sync_fn:
               Callback that performs the allgather operation on the metric state. When ``None``, DDP
               will be used to perform the all gather.

       Example:
           >>> from torchmetrics import CosineSimilarity
           >>> target = torch.tensor([[0, 1], [1, 1]])
           >>> preds = torch.tensor([[0, 1], [0, 1]])
           >>> cosine_similarity = CosineSimilarity()
           >>> cosine_similarity.update(preds, target)
           >>> cosine_similarity.compute()
           tensor([1.0000, 1.0000, 1.0000])
    """

    def __init__(
        self,
        reduction: str = 'sum',
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("correct", default=tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")
        self.reduction = reduction

    def update(self, preds: Tensor, target: Tensor):
        """
        Update state with predictions and targets. See :ref:`references/modules:input types` for more information
        on input types.

        Args:
            preds: Predictions from model (probabilities, logits or labels)
            target: Ground truth labels
        """
        correct, total = _cosine_similarity_update(preds, target)

        self.correct += correct
        self.total += total

    def compute(self):
        return _cosine_similarity_compute(self.total, self.correct, self.reduction)
