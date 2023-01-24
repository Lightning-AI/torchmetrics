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
from torch import Tensor, tensor

from torchmetrics.functional.retrieval.average_precision import retrieval_average_precision
from torchmetrics.retrieval.base import RetrievalMetric


class RetrievalMAP(RetrievalMetric):
    """Computes `Mean Average Precision`_.

    Works with binary target data. Accepts float predictions from a model output.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): A float tensor of shape ``(N, ...)``
    - ``target`` (:class:`~torch.Tensor`): A long or bool tensor of shape ``(N, ...)``
    - ``indexes`` (:class:`~torch.Tensor`): A long tensor of shape ``(N, ...)`` which indicate to which query a
      prediction belongs

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``rmap`` (:class:`~torch.Tensor`): A tensor with the mean average precision of the predictions ``preds``
      w.r.t. the labels ``target``

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

        ignore_index:
            Ignore predictions where the target is equal to this number.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``empty_target_action`` is not one of ``error``, ``skip``, ``neg`` or ``pos``.
        ValueError:
            If ``ignore_index`` is not `None` or an integer.

    Example:
        >>> from torchmetrics import RetrievalMAP
        >>> indexes = tensor([0, 0, 0, 1, 1, 1, 1])
        >>> preds = tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
        >>> target = tensor([False, False, True, False, True, False, True])
        >>> rmap = RetrievalMAP()
        >>> rmap(preds, target, indexes=indexes)
        tensor(0.7917)
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def _metric(self, preds: Tensor, target: Tensor) -> Tensor:
        return retrieval_average_precision(preds, target)
