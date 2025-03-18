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
from collections.abc import Sequence
from typing import Any, Callable, Optional, Union

from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.retrieval.hit_rate import retrieval_hit_rate
from torchmetrics.retrieval.base import RetrievalMetric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["RetrievalHitRate.plot"]


class RetrievalHitRate(RetrievalMetric):
    """Compute `IR HitRate`.

    Works with binary target data. Accepts float predictions from a model output.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): A float tensor of shape ``(N, ...)``
    - ``target`` (:class:`~torch.Tensor`): A long or bool tensor of shape ``(N, ...)``
    - ``indexes`` (:class:`~torch.Tensor`): A long tensor of shape ``(N, ...)`` which indicate to which query a
      prediction belongs

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``hr@k`` (:class:`~torch.Tensor`): A single-value tensor with the hit rate (at ``top_k``) of the predictions
      ``preds`` w.r.t. the labels ``target``

    All ``indexes``, ``preds`` and ``target`` must have the same dimension and will be flatten at the beginning,
    so that for example, a tensor of shape ``(N, M)`` is treated as ``(N * M, )``. Predictions will be first grouped by
    ``indexes`` and then will be computed as the mean of the metric over each query.


    Args:
        empty_target_action:
            Specify what to do with queries that do not have at least a positive ``target``. Choose from:

            - ``'neg'``: those queries count as ``0.0`` (default)
            - ``'pos'``: those queries count as ``1.0``
            - ``'skip'``: skip those queries; if all queries are skipped, ``0.0`` is returned
            - ``'error'``: raise a ``ValueError``

        ignore_index: Ignore predictions where the target is equal to this number.
        top_k: Consider only the top k elements for each query (default: ``None``, which considers them all)
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
        ValueError:
            If ``top_k`` is not ``None`` or not an integer greater than 0.

    Example:
        >>> from torch import tensor
        >>> from torchmetrics.retrieval import RetrievalHitRate
        >>> indexes = tensor([0, 0, 0, 1, 1, 1, 1])
        >>> preds = tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
        >>> target = tensor([True, False, False, False, True, False, True])
        >>> hr2 = RetrievalHitRate(top_k=2)
        >>> hr2(preds, target, indexes=indexes)
        tensor(0.5000)

    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def __init__(
        self,
        empty_target_action: str = "neg",
        ignore_index: Optional[int] = None,
        top_k: Optional[int] = None,
        aggregation: Union[Literal["mean", "median", "min", "max"], Callable] = "mean",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            empty_target_action=empty_target_action,
            ignore_index=ignore_index,
            aggregation=aggregation,
            **kwargs,
        )

        if top_k is not None and not (isinstance(top_k, int) and top_k > 0):
            raise ValueError("`top_k` has to be a positive integer or None")
        self.top_k = top_k

    def _metric(self, preds: Tensor, target: Tensor) -> Tensor:
        return retrieval_hit_rate(preds, target, top_k=self.top_k)

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
            >>> from torchmetrics.retrieval import RetrievalHitRate
            >>> # Example plotting a single value
            >>> metric = RetrievalHitRate()
            >>> metric.update(torch.rand(10,), torch.randint(2, (10,)), indexes=torch.randint(2,(10,)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> import torch
            >>> from torchmetrics.retrieval import RetrievalHitRate
            >>> # Example plotting multiple values
            >>> metric = RetrievalHitRate()
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(torch.rand(10,), torch.randint(2, (10,)), indexes=torch.randint(2,(10,))))
            >>> fig, ax = metric.plot(values)

        """
        return self._plot(val, ax)
