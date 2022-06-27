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
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import torch
from torch import Tensor, tensor

from torchmetrics import Metric
from torchmetrics.utilities.checks import _check_retrieval_inputs
from torchmetrics.utilities.data import get_group_indexes

#: get_group_indexes is used to group predictions belonging to the same document


class RetrievalMetric(Metric, ABC):
    """Works with binary target data. Accepts float predictions from a model output.

    Forward accepts

    - ``preds`` (float tensor): ``(N, ...)``
    - ``target`` (long or bool tensor): ``(N, ...)``
    - ``indexes`` (long tensor): ``(N, ...)``

    ``indexes``, ``preds`` and ``target`` must have the same dimension and will be flatten
    to single dimension once provided.

    ``indexes`` indicate to which query a prediction belongs.
    Predictions will be first grouped by indexes. Then the
    real metric, defined by overriding the `_metric` method,
    will be computed as the mean of the scores over each query.

    Args:
        empty_target_action:
            Specify what to do with queries that do not have at least a positive
            or negative (depend on metric) target. Choose from:

            - ``'neg'``: those queries count as ``0.0`` (default)
            - ``'pos'``: those queries count as ``1.0``
            - ``'skip'``: skip those queries; if all queries are skipped, ``0.0`` is returned
            - ``'error'``: raise a ``ValueError``

        ignore_index:
            Ignore predictions where the target is equal to this number.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``empty_target_action`` is not one of ``error``, ``skip``, ``neg`` or ``pos``.
        ValueError:
            If ``ignore_index`` is not `None` or an integer.
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    indexes: List[Tensor]
    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
        self,
        empty_target_action: str = "neg",
        ignore_index: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.allow_non_binary_target = False

        empty_target_action_options = ("error", "skip", "neg", "pos")
        if empty_target_action not in empty_target_action_options:
            raise ValueError(f"Argument `empty_target_action` received a wrong value `{empty_target_action}`.")

        self.empty_target_action = empty_target_action

        if ignore_index is not None and not isinstance(ignore_index, int):
            raise ValueError("Argument `ignore_index` must be an integer or None.")

        self.ignore_index = ignore_index

        self.add_state("indexes", default=[], dist_reduce_fx=None)
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("target", default=[], dist_reduce_fx=None)

    def update(self, preds: Tensor, target: Tensor, indexes: Tensor) -> None:  # type: ignore
        """Check shape, check and convert dtypes, flatten and add to accumulators."""
        if indexes is None:
            raise ValueError("Argument `indexes` cannot be None")

        indexes, preds, target = _check_retrieval_inputs(
            indexes, preds, target, allow_non_binary_target=self.allow_non_binary_target, ignore_index=self.ignore_index
        )

        self.indexes.append(indexes)
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> Tensor:
        """First concat state ``indexes``, ``preds`` and ``target`` since they were stored as lists.

        After that, compute list of groups that will help in keeping together predictions about the same query. Finally,
        for each group compute the ``_metric`` if the number of positive targets is at least 1, otherwise behave as
        specified by ``self.empty_target_action``.
        """
        indexes = torch.cat(self.indexes, dim=0)
        preds = torch.cat(self.preds, dim=0)
        target = torch.cat(self.target, dim=0)

        res = []
        groups = get_group_indexes(indexes)

        for group in groups:
            mini_preds = preds[group]
            mini_target = target[group]

            if not mini_target.sum():
                if self.empty_target_action == "error":
                    raise ValueError("`compute` method was provided with a query with no positive target.")
                if self.empty_target_action == "pos":
                    res.append(tensor(1.0))
                elif self.empty_target_action == "neg":
                    res.append(tensor(0.0))
            else:
                # ensure list contains only float tensors
                res.append(self._metric(mini_preds, mini_target))

        return torch.stack([x.to(preds) for x in res]).mean() if res else tensor(0.0).to(preds)

    @abstractmethod
    def _metric(self, preds: Tensor, target: Tensor) -> Tensor:
        """Compute a metric over a predictions and target of a single group.

        This method should be overridden by subclasses.
        """
