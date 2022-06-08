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
from typing import Any, Optional, Tuple

import torch
from torch import Tensor, tensor

from torchmetrics import Metric
from torchmetrics.functional.retrieval.precision_recall_curve import retrieval_precision_recall_curve
from torchmetrics.utilities.checks import _check_retrieval_inputs
from torchmetrics.utilities.data import dim_zero_cat, get_group_indexes


def _retrieval_recall_at_fixed_precision(
    precision: Tensor,
    recall: Tensor,
    top_k: Tensor,
    min_precision: float,
) -> Tuple[Tensor, Tensor]:
    """Computes maximum recall with condition that corresponding precision >= `min_precision`.

    Args:
        top_k: tensor with all possible k
        precision: tensor with all values precisions@k for k from top_k tensor
        recall: tensor with all values recall@k for k from top_k tensor
        min_precision: float value specifying minimum precision threshold.

    Returns:
        Maximum recall value, corresponding it best k
    """
    try:
        max_recall, best_k = max((r, k) for p, r, k in zip(precision, recall, top_k) if p >= min_precision)

    except ValueError:
        max_recall = torch.tensor(0.0, device=recall.device, dtype=recall.dtype)
        best_k = torch.tensor(len(top_k))

    if max_recall == 0.0:
        best_k = torch.tensor(len(top_k), device=top_k.device, dtype=top_k.dtype)

    return max_recall, best_k


class RetrievalPrecisionRecallCurve(Metric):
    """Computes precision-recall pairs for different k (from 1 to `max_k`).

    In a ranked retrieval context, appropriate sets of retrieved documents are naturally given by
    the top k retrieved documents.

    Recall is the fraction of relevant documents retrieved among all the relevant documents.
    Precision is the fraction of relevant documents among all the retrieved documents.

    For each such set, precision and recall values can be plotted to give a recall-precision
    curve.

    Forward accepts:

    - ``preds`` (float tensor): ``(N, ...)``
    - ``target`` (long or bool tensor): ``(N, ...)``
    - ``indexes`` (long tensor): ``(N, ...)``

    ``indexes``, ``preds`` and ``target`` must have the same dimension.
    ``indexes`` indicate to which query a prediction belongs.
    Predictions will be first grouped by ``indexes`` and then `RetrievalRecallAtFixedPrecision`
    will be computed as the mean of the `RetrievalRecallAtFixedPrecision` over each query.

    Args:
        max_k: Calculate recall and precision for all possible top k from 1 to max_k
               (default: `None`, which considers all possible top k)
        adaptive_k: adjust `k` to `min(k, number of documents)` for each query
        empty_target_action:
            Specify what to do with queries that do not have at least a positive ``target``. Choose from:

            - ``'neg'``: those queries count as ``0.0`` (default)
            - ``'pos'``: those queries count as ``1.0``
            - ``'skip'``: skip those queries; if all queries are skipped, ``0.0`` is returned
            - ``'error'``: raise a ``ValueError``

        ignore_index:
            Ignore predictions where the target is equal to this number.
        kwargs:
            Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``empty_target_action`` is not one of ``error``, ``skip``, ``neg`` or ``pos``.
        ValueError:
            If ``ignore_index`` is not `None` or an integer.
        ValueError:
            If ``max_k`` parameter is not `None` or an integer larger than 0.

    Example:
        >>> from torchmetrics import RetrievalPrecisionRecallCurve
        >>> indexes = tensor([0, 0, 0, 0, 1, 1, 1])
        >>> preds = tensor([0.4, 0.01, 0.5, 0.6, 0.2, 0.3, 0.5])
        >>> target = tensor([True, False, False, True, True, False, True])
        >>> r = RetrievalPrecisionRecallCurve(max_k=4)
        >>> precisions, recalls, top_k = r(preds, target, indexes=indexes)
        >>> precisions
        tensor([1.0000, 0.5000, 0.6667, 0.5000])
        >>> recalls
        tensor([0.5000, 0.5000, 1.0000, 1.0000])
        >>> top_k
        tensor([1, 2, 3, 4])
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        max_k: Optional[int] = None,
        adaptive_k: bool = False,
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

        if (max_k is not None) and not (isinstance(max_k, int) and max_k > 0):
            raise ValueError("`max_k` has to be a positive integer or None")
        self.max_k = max_k

        if not isinstance(adaptive_k, bool):
            raise ValueError("`adaptive_k` has to be a boolean")
        self.adaptive_k = adaptive_k

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

    def compute(self) -> Tuple[Tensor, Tensor, Tensor]:
        # concat all data
        indexes = dim_zero_cat(self.indexes)
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        groups = get_group_indexes(indexes)

        # don't want to change self.max_k
        max_k = self.max_k
        if max_k is None:
            # set max_k as size of max group by size
            max_k = max(map(len, groups))

        precisions, recalls = [], []

        for group in groups:
            mini_preds = preds[group]
            mini_target = target[group]

            if not mini_target.sum():
                if self.empty_target_action == "error":
                    raise ValueError("`compute` method was provided with a query with no positive target.")
                elif self.empty_target_action == "pos":
                    recalls.append(torch.ones(max_k, device=preds.device))
                    precisions.append(torch.ones(max_k, device=preds.device))
                elif self.empty_target_action == "neg":
                    recalls.append(torch.zeros(max_k, device=preds.device))
                    precisions.append(torch.zeros(max_k, device=preds.device))
            else:
                precision, recall, _ = retrieval_precision_recall_curve(mini_preds, mini_target, max_k, self.adaptive_k)

                precisions.append(precision)
                recalls.append(recall)

        precision = (
            torch.stack([x.to(preds) for x in precisions]).mean(dim=0) if precisions else torch.zeros(max_k).to(preds)
        )
        recall = torch.stack([x.to(preds) for x in recalls]).mean(dim=0) if recalls else torch.zeros(max_k).to(preds)
        top_k = torch.arange(1, max_k + 1, device=preds.device)

        return precision, recall, top_k


class RetrievalRecallAtFixedPrecision(RetrievalPrecisionRecallCurve):
    """Computes `IR Recall at fixed Precision`_.

    Forward accepts:

    - ``preds`` (float tensor): ``(N, ...)``
    - ``target`` (long or bool tensor): ``(N, ...)``
    - ``indexes`` (long tensor): ``(N, ...)``

    ``indexes``, ``preds`` and ``target`` must have the same dimension.
    ``indexes`` indicate to which query a prediction belongs.
    Predictions will be first grouped by ``indexes`` and then `RetrievalRecallAtFixedPrecision`
    will be computed as the mean of the `RetrievalRecallAtFixedPrecision` over each query.

    Args:
        min_precision: float value specifying minimum precision threshold.
        max_k: Calculate recall and precision for all possible top k from 1 to max_k
               (default: `None`, which considers all possible top k)
        adaptive_k: adjust `k` to `min(k, number of documents)` for each query
        empty_target_action:
            Specify what to do with queries that do not have at least a positive ``target``. Choose from:

            - ``'neg'``: those queries count as ``0.0`` (default)
            - ``'pos'``: those queries count as ``1.0``
            - ``'skip'``: skip those queries; if all queries are skipped, ``0.0`` is returned
            - ``'error'``: raise a ``ValueError``

        ignore_index:
            Ignore predictions where the target is equal to this number.
        kwargs:
            Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``empty_target_action`` is not one of ``error``, ``skip``, ``neg`` or ``pos``.
        ValueError:
            If ``ignore_index`` is not `None` or an integer.
        ValueError:
            If ``min_precision`` parameter is not float or between 0 and 1.
        ValueError:
            If ``max_k`` parameter is not `None` or an integer larger than 0.

    Example:
        >>> from torchmetrics import RetrievalRecallAtFixedPrecision
        >>> indexes = tensor([0, 0, 0, 0, 1, 1, 1])
        >>> preds = tensor([0.4, 0.01, 0.5, 0.6, 0.2, 0.3, 0.5])
        >>> target = tensor([True, False, False, True, True, False, True])
        >>> r = RetrievalRecallAtFixedPrecision(min_precision=0.8)
        >>> r(preds, target, indexes=indexes)
        (tensor(0.5000), tensor(1))
    """

    higher_is_better = True

    def __init__(
        self,
        min_precision: float = 0.0,
        max_k: Optional[int] = None,
        adaptive_k: bool = False,
        empty_target_action: str = "neg",
        ignore_index: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            max_k=max_k,
            adaptive_k=adaptive_k,
            empty_target_action=empty_target_action,
            ignore_index=ignore_index,
            **kwargs,
        )

        if not (isinstance(min_precision, float) and 0.0 <= min_precision <= 1.0):
            raise ValueError("`min_precision` has to be a positive float between 0 and 1")

        self.min_precision = min_precision

    def compute(self) -> Tuple[Tensor, Tensor]:  # type: ignore
        precisions, recalls, top_k = super().compute()

        return _retrieval_recall_at_fixed_precision(precisions, recalls, top_k, self.min_precision)
