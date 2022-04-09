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
from typing import Any, Dict, Optional

from torch import Tensor, tensor

from torchmetrics.functional.retrieval.recall_precision import retrieval_recall_at_precision
from torchmetrics.retrieval.base import RetrievalMetric


class RetrievalRecallAtFixedPrecision(RetrievalMetric):
    """
    Computes `IR Recall at fixed Precision`_.

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
        max_k:
            Calculate recall and precision for all possible top k from 0 to max_k
            (default: `None`, which considers all possible top k)
        empty_target_action:
            Specify what to do with queries that do not have at least a positive ``target``. Choose from:

            - ``'neg'``: those queries count as ``0.0`` (default)
            - ``'pos'``: those queries count as ``1.0``
            - ``'skip'``: skip those queries; if all queries are skipped, ``0.0`` is returned
            - ``'error'``: raise a ``ValueError``

        ignore_index:
            Ignore predictions where the target is equal to this number.
        compute_on_step:
            Forward only calls ``update()`` and returns None if this is set to False.

            .. deprecated:: v0.8
                Argument has no use anymore and will be removed v0.9.

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
        tensor(0.5625)
    """

    higher_is_better = True

    def __init__(
        self,
        min_precision: float,
        max_k: Optional[int] = None,
        empty_target_action: str = "neg",
        ignore_index: Optional[int] = None,
        compute_on_step: Optional[bool] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(
            empty_target_action=empty_target_action,
            ignore_index=ignore_index,
            compute_on_step=compute_on_step,
            **kwargs,
        )

        if (max_k is not None) and not (isinstance(max_k, int) and max_k > 0):
            raise ValueError("`max_k` has to be a positive integer or None")

        if not(isinstance(min_precision, float) and 0. <= min_precision <= 1.):
            raise ValueError("`min_precision` has to be a positive float between 0 and 1")

        self.max_k = max_k
        self.min_precision = min_precision

    def _metric(self, preds: Tensor, target: Tensor) -> Tensor:
        return retrieval_recall_at_precision(
            preds, target, min_precision=self.min_precision, max_k=self.max_k
        )
