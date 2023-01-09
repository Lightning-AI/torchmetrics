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

from torchmetrics.functional.retrieval.r_precision import retrieval_r_precision
from torchmetrics.retrieval.base import RetrievalMetric


class RetrievalRPrecision(RetrievalMetric):
    """Computes `IR R-Precision`_.

    Works with binary target data. Accepts float predictions from a model output.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): A float tensor of shape ``(N, ...)``
    - ``target`` (:class:`~torch.Tensor`): A long or bool tensor of shape ``(N, ...)``
    - ``indexes`` (:class:`~torch.Tensor`): A long tensor of shape ``(N, ...)`` which indicate to which query a
      prediction belongs

    .. note:: All ``indexes``, ``preds`` and ``target`` must have the same dimension.

    .. note::
        Predictions will be first grouped by ``indexes`` and then `R-Precision` will be computed as the mean
        of the `R-Precision` over each query.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``p2`` (:class:`~torch.Tensor`): A single-value tensor with the r-precision of the predictions ``preds``
      w.r.t. the labels ``target``.

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
        >>> from torchmetrics import RetrievalRPrecision
        >>> indexes = tensor([0, 0, 0, 1, 1, 1, 1])
        >>> preds = tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
        >>> target = tensor([False, False, True, False, True, False, True])
        >>> p2 = RetrievalRPrecision()
        >>> p2(preds, target, indexes=indexes)
        tensor(0.7500)
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def _metric(self, preds: Tensor, target: Tensor) -> Tensor:
        return retrieval_r_precision(preds, target)
