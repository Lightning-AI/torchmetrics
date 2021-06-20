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


class CosineSimilarty(Metric):
    r"""
       Computes the `Cosine Similarity <https://en.wikipedia.org/wiki/Cosine_similarity>`_
        between targets and predictions:
       Accepts all input types listed in :ref:`references/modules:input types`.
       Args:
           threshold:
               Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case
               of binary or multi-label inputs. Default value of 0.5 corresponds to input being probabilities.
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
           reduction: The method of reducing along the batch dimension using sum, mean or
                        taking the individual scores
       Raises:
           ValueError:
               If ``threshold`` is not between ``0`` and ``1``.
       Example:
           >>> from torchmetrics import CosineSimilarity
           >>> target = torch.tensor([[0, 1], [1, 1]])
           >>> preds = torch.tensor([[0, 1], [0, 1]])
           >>> cosine_similarity = CosineSimilarity()
           >>> cosine_similarity.update(preds, target, 'none')
           >>> cosine_similarity.compute()
           tensor([1.0000, 1.0000, 1.0000])
    """

    def __init__(
        self,
        threshold: float = 0.5,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
        reduction: str = 'sum'
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

        if not 0 < threshold < 1:
            raise ValueError("The `threshold` should lie in the (0,1) interval.")
        self.threshold = threshold

    def update(self, preds: Tensor, target: Tensor, reduction="sum"):
        """
        Update state with predictions and targets. See :ref:`references/modules:input types` for more information
        on input types.
        Args:
            preds: Predictions from model (probabilities, logits or labels)
            target: Ground truth labels
            reduction : how to reduce over the batch dimension using sum, mean or
                        taking the individual scores
        """
        correct, total = _cosine_similarity_update(preds, target)

        self.correct += correct
        self.total += total
        self.reduction = reduction

    def compute(self):
        return _cosine_similarity_compute(self.total, self.correct, self.reduction)
