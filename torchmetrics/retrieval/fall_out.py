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
from typing import Any, Optional

import torch
from torch import Tensor, tensor

from torchmetrics.functional.retrieval.fall_out import retrieval_fall_out
from torchmetrics.retrieval.base import RetrievalMetric
from torchmetrics.utilities.data import get_group_indexes


class RetrievalFallOut(RetrievalMetric):
    """Computes `Fall-out`_.

    Works with binary target data. Accepts float predictions from a model output.

    Forward accepts:

    - ``preds`` (float tensor): ``(N, ...)``
    - ``target`` (long or bool tensor): ``(N, ...)``
    - ``indexes`` (long tensor): ``(N, ...)``

    ``indexes``, ``preds`` and ``target`` must have the same dimension.
    ``indexes`` indicate to which query a prediction belongs.
    Predictions will be first grouped by ``indexes`` and then `Fall-out` will be computed as the mean
    of the `Fall-out` over each query.

    Args:
        empty_target_action:
            Specify what to do with queries that do not have at least a negative ``target``. Choose from:

            - ``'neg'``: those queries count as ``0.0`` (default)
            - ``'pos'``: those queries count as ``1.0``
            - ``'skip'``: skip those queries; if all queries are skipped, ``0.0`` is returned
            - ``'error'``: raise a ``ValueError``

        ignore_index:
            Ignore predictions where the target is equal to this number.
        k: consider only the top k elements for each query (default: `None`, which considers them all)
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``empty_target_action`` is not one of ``error``, ``skip``, ``neg`` or ``pos``.
        ValueError:
            If ``ignore_index`` is not `None` or an integer.
        ValueError:
            If ``k`` parameter is not `None` or an integer larger than 0.

    Example:
        >>> from torchmetrics import RetrievalFallOut
        >>> indexes = tensor([0, 0, 0, 1, 1, 1, 1])
        >>> preds = tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
        >>> target = tensor([False, False, True, False, True, False, True])
        >>> fo = RetrievalFallOut(k=2)
        >>> fo(preds, target, indexes=indexes)
        tensor(0.5000)
    """

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(
        self,
        empty_target_action: str = "pos",
        ignore_index: Optional[int] = None,
        k: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            empty_target_action=empty_target_action,
            ignore_index=ignore_index,
            **kwargs,
        )

        if (k is not None) and not (isinstance(k, int) and k > 0):
            raise ValueError("`k` has to be a positive integer or None")
        self.k = k

    def compute(self) -> Tensor:
        """First concat state ``indexes``, ``preds`` and ``target`` since they were stored as lists.

        After that, compute list of groups that will help in keeping together predictions about the same query. Finally,
        for each group compute the `_metric` if the number of negative targets is at least 1, otherwise behave as
        specified by `self.empty_target_action`.
        """
        indexes = torch.cat(self.indexes, dim=0)
        preds = torch.cat(self.preds, dim=0)
        target = torch.cat(self.target, dim=0)

        res = []
        groups = get_group_indexes(indexes)

        for group in groups:
            mini_preds = preds[group]
            mini_target = target[group]

            if not (1 - mini_target).sum():
                if self.empty_target_action == "error":
                    raise ValueError("`compute` method was provided with a query with no negative target.")
                if self.empty_target_action == "pos":
                    res.append(tensor(1.0))
                elif self.empty_target_action == "neg":
                    res.append(tensor(0.0))
            else:
                # ensure list containt only float tensors
                res.append(self._metric(mini_preds, mini_target))

        return torch.stack([x.to(preds) for x in res]).mean() if res else tensor(0.0).to(preds)

    def _metric(self, preds: Tensor, target: Tensor) -> Tensor:
        return retrieval_fall_out(preds, target, k=self.k)
