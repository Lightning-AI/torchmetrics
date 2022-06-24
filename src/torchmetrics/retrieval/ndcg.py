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

from torch import Tensor, tensor

from torchmetrics.functional.retrieval.ndcg import retrieval_normalized_dcg
from torchmetrics.retrieval.base import RetrievalMetric


class RetrievalNormalizedDCG(RetrievalMetric):
    """Computes `Normalized Discounted Cumulative Gain`_.

    Works with binary or positive integer target data. Accepts float predictions from a model output.

    Forward accepts:

    - ``preds`` (float tensor): ``(N, ...)``
    - ``target`` (long, int, bool or float tensor): ``(N, ...)``
    - ``indexes`` (long tensor): ``(N, ...)``

    ``indexes``, ``preds`` and ``target`` must have the same dimension.
    ``indexes`` indicate to which query a prediction belongs.
    Predictions will be first grouped by ``indexes`` and then `Normalized Discounted Cumulative Gain`
    will be computed as the mean of the `Normalized Discounted Cumulative Gain` over each query.

    Args:
        empty_target_action:
            Specify what to do with queries that do not have at least a positive ``target``. Choose from:

            - ``'neg'``: those queries count as ``0.0`` (default)
            - ``'pos'``: those queries count as ``1.0``
            - ``'skip'``: skip those queries; if all queries are skipped, ``0.0`` is returned
            - ``'error'``: raise a ``ValueError``

        ignore_index:
            Ignore predictions where the target is equal to this number.
        k: consider only the top k elements for each query (default: ``None``, which considers them all)
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``empty_target_action`` is not one of ``error``, ``skip``, ``neg`` or ``pos``.
        ValueError:
            If ``ignore_index`` is not `None` or an integer.
        ValueError:
            If ``k`` parameter is not `None` or an integer larger than 0.

    Example:
        >>> from torchmetrics import RetrievalNormalizedDCG
        >>> indexes = tensor([0, 0, 0, 1, 1, 1, 1])
        >>> preds = tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
        >>> target = tensor([False, False, True, False, True, False, True])
        >>> ndcg = RetrievalNormalizedDCG()
        >>> ndcg(preds, target, indexes=indexes)
        tensor(0.8467)
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        empty_target_action: str = "neg",
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
        self.allow_non_binary_target = True

    def _metric(self, preds: Tensor, target: Tensor) -> Tensor:
        return retrieval_normalized_dcg(preds, target, k=self.k)
