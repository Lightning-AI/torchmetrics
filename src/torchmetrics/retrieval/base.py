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
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Union

import torch
from torch import Tensor, tensor
from typing_extensions import Literal

from torchmetrics import Metric
from torchmetrics.utilities.checks import _check_retrieval_inputs
from torchmetrics.utilities.data import _flexible_bincount, dim_zero_cat


def _retrieval_aggregate(
    values: Tensor,
    aggregation: Union[Literal["mean", "median", "min", "max"], Callable] = "mean",
    dim: Optional[int] = None,
) -> Tensor:
    """Aggregate the final retrieval values into a single value."""
    if aggregation == "mean":
        return values.mean() if dim is None else values.mean(dim=dim)
    if aggregation == "median":
        return values.median() if dim is None else values.median(dim=dim).values
    if aggregation == "min":
        return values.min() if dim is None else values.min(dim=dim).values
    if aggregation == "max":
        return values.max() if dim is None else values.max(dim=dim).values
    return aggregation(values, dim=dim)


class RetrievalMetric(Metric, ABC):
    """Works with binary target data. Accepts float predictions from a model output.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): A float tensor of shape ``(N, ...)``
    - ``target`` (:class:`~torch.Tensor`): A long or bool tensor of shape ``(N, ...)``
    - ``indexes`` (:class:`~torch.Tensor`): A long tensor of shape ``(N, ...)`` which indicate to which query a
      prediction belongs

    .. hint::
        The ``indexes``, ``preds`` and ``target`` must have the same dimension and will be flattened
        to single dimension once provided.

    .. attention::
        Predictions will be first grouped by ``indexes`` and then the real metric, defined by overriding
        the `_metric` method, will be computed as the mean of the scores over each query.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``metric`` (:class:`~torch.Tensor`): A tensor as computed by ``_metric`` if the number of positive targets is
      at least 1, otherwise behave as specified by ``self.empty_target_action``.

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
        aggregation:
            Specify how to aggregate over indexes. Can either a custom callable function that takes in a single tensor
            and returns a scalar value or one of the following strings:

            - ``'mean'``: average value is returned
            - ``'median'``: median value is returned
            - ``'max'``: max value is returned
            - ``'min'``: min value is returned

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
        aggregation: Union[Literal["mean", "median", "min", "max"], Callable] = "mean",
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

        if not (aggregation in ("mean", "median", "min", "max") or callable(aggregation)):
            raise ValueError(
                "Argument `aggregation` must be one of `mean`, `median`, `min`, `max` or a custom callable function"
                f"which takes tensor of values, but got {aggregation}."
            )
        self.aggregation = aggregation

        self.add_state("indexes", default=[], dist_reduce_fx=None)
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("target", default=[], dist_reduce_fx=None)

    def update(self, preds: Tensor, target: Tensor, indexes: Tensor) -> None:
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
        indexes = dim_zero_cat(self.indexes)
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)

        indexes, indices = torch.sort(indexes)
        preds = preds[indices]
        target = target[indices]

        split_sizes = _flexible_bincount(indexes).detach().cpu().tolist()

        res = []
        for mini_preds, mini_target in zip(
            torch.split(preds, split_sizes, dim=0), torch.split(target, split_sizes, dim=0)
        ):
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

        if res:
            return _retrieval_aggregate(torch.stack([x.to(preds) for x in res]), self.aggregation)
        return tensor(0.0).to(preds)

    @abstractmethod
    def _metric(self, preds: Tensor, target: Tensor) -> Tensor:
        """Compute a metric over a predictions and target of a single group.

        This method should be overridden by subclasses.

        """
