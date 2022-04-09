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
from typing import Optional

from torch import Tensor, tensor

from torchmetrics.functional.retrieval import retrieval_recall, retrieval_precision
from torchmetrics.utilities.checks import _check_retrieval_functional_inputs


def retrieval_recall_at_precision(
    preds: Tensor,
    target: Tensor,
    min_precision: float,
    max_k: Optional[int] = None
) -> Tensor:
    """Computes the recall at fixed precision metric (for information retrieval).
    Recall at Fixed Precision is maximum possible Recall with some Precision threshold.
    (for all possible top k)

    ``preds`` and ``target`` should be of the same shape and live on the same device. If no ``target`` is ``True``,
    ``0`` is returned. ``target`` must be either `bool` or `integers` and ``preds`` must be `float`,
    otherwise an error is raised. If you want to measure Recall@K, ``k`` must be a positive integer.

    Args:
        preds: estimated probabilities of each document to be relevant.
        target: ground truth about each document being relevant or not.
        min_precision: float value specifying minimum precision threshold.
        max_k:
            Calculate recall and precision for all possible top k from 0 to max_k
            (default: `None`, which considers all possible top k)

    Returns:
        a single-value tensor with the recall at fixed precision of the predictions ``preds``
        w.r.t. the labels ``target``.

    Raises:
        ValueError:
            If ``min_precision`` parameter is not float or between 0 and 1.
        ValueError:
            If ``max_k`` parameter is not `None` or an integer larger than 0.

    Example:
        >>> from  torchmetrics.functional import retrieval_recall_at_precision
        >>> preds = tensor([0.4, 0.01, 0.5, 0.6])
        >>> target = tensor([True, False, False, True])
        >>> retrieval_recall_at_precision(preds, target, min_precision=0.8)
        tensor(0.5000)
    """
    preds, target = _check_retrieval_functional_inputs(preds, target)

    if not(isinstance(min_precision, float) and 0. <= min_precision <= 1.):
        raise ValueError("`min_precision` has to be a positive float between 0 and 1")

    if max_k is None:
        max_k = preds.shape[-1]

    if not (isinstance(max_k, int) and max_k > 0):
        raise ValueError("`max_k` has to be a positive integer or None")

    if not target.sum():
        return tensor(0.0, device=preds.device)

    recall = []
    precision = []

    for k in range(1, max_k + 1):
        recall.append(retrieval_recall(preds, target, k=k))
        precision.append(retrieval_precision(preds, target, k=k))

    max_recall, _ = max(
        (r, p) for p, r in zip(precision, recall) if p >= min_precision
    )

    return max_recall
