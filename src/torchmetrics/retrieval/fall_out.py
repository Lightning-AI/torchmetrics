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
from typing import Any, Optional, Sequence, Union

import torch
from torch import Tensor, tensor

from torchmetrics.functional.retrieval.fall_out import retrieval_fall_out
from torchmetrics.retrieval.base import RetrievalMetric
from torchmetrics.utilities.data import _flexible_bincount, dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["RetrievalFallOut.plot"]


class RetrievalFallOut(RetrievalMetric):
    """Compute `Fall-out`_.

    Works with binary target data. Accepts float predictions from a model output.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): A float tensor of shape ``(N, ...)``
    - ``target`` (:class:`~torch.Tensor`): A long or bool tensor of shape ``(N, ...)``
    - ``indexes`` (:class:`~torch.Tensor`): A long tensor of shape ``(N, ...)`` which indicate to which query a
      prediction belongs

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``fo`` (:class:`~torch.Tensor`): A tensor with the computed metric

    All ``indexes``, ``preds`` and ``target`` must have the same dimension and will be flatten at the beginning,
    so that for example, a tensor of shape ``(N, M)`` is treated as ``(N * M, )``. Predictions will be first grouped by
    ``indexes`` and then will be computed as the mean of the metric over each query.

    Args:
        empty_target_action:
            Specify what to do with queries that do not have at least a negative ``target``. Choose from:

            - ``'neg'``: those queries count as ``0.0`` (default)
            - ``'pos'``: those queries count as ``1.0``
            - ``'skip'``: skip those queries; if all queries are skipped, ``0.0`` is returned
            - ``'error'``: raise a ``ValueError``

        ignore_index:
            Ignore predictions where the target is equal to this number.
        top_k: consider only the top k elements for each query (default: `None`, which considers them all)
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``empty_target_action`` is not one of ``error``, ``skip``, ``neg`` or ``pos``.
        ValueError:
            If ``ignore_index`` is not `None` or an integer.
        ValueError:
            If ``top_k`` parameter is not `None` or an integer larger than 0.

    Example:
        >>> from torchmetrics.retrieval import RetrievalFallOut
        >>> indexes = tensor([0, 0, 0, 1, 1, 1, 1])
        >>> preds = tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
        >>> target = tensor([False, False, True, False, True, False, True])
        >>> fo = RetrievalFallOut(top_k=2)
        >>> fo(preds, target, indexes=indexes)
        tensor(0.5000)
    """

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def __init__(
        self,
        empty_target_action: str = "pos",
        ignore_index: Optional[int] = None,
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            empty_target_action=empty_target_action,
            ignore_index=ignore_index,
            **kwargs,
        )

        if top_k is not None and not (isinstance(top_k, int) and top_k > 0):
            raise ValueError("`top_k` has to be a positive integer or None")
        self.top_k = top_k

    def compute(self) -> Tensor:
        """First concat state ``indexes``, ``preds`` and ``target`` since they were stored as lists.

        After that, compute list of groups that will help in keeping together predictions about the same query. Finally,
        for each group compute the `_metric` if the number of negative targets is at least 1, otherwise behave as
        specified by `self.empty_target_action`.
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
        return retrieval_fall_out(preds, target, top_k=self.top_k)

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> import torch
            >>> from torchmetrics.retrieval import RetrievalFallOut
            >>> # Example plotting a single value
            >>> metric = RetrievalFallOut()
            >>> metric.update(torch.rand(10,), torch.randint(2, (10,)), indexes=torch.randint(2,(10,)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> import torch
            >>> from torchmetrics.retrieval import RetrievalFallOut
            >>> # Example plotting multiple values
            >>> metric = RetrievalFallOut()
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(torch.rand(10,), torch.randint(2, (10,)), indexes=torch.randint(2,(10,))))
            >>> fig, ax = metric.plot(values)
        """
        return self._plot(val, ax)
