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
from typing import Optional

import torch
from torch import Tensor

from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.compute import _safe_divide


def _critical_success_index_update(
    preds: Tensor, target: Tensor, threshold: float, keep_sequence_dim: Optional[int] = None
) -> tuple[Tensor, Tensor, Tensor]:
    """Update and return variables required to compute Critical Success Index. Checks for same shape of tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        threshold: Values above or equal to threshold are replaced with 1, below by 0
        keep_sequence_dim: Index of the sequence dimension if the inputs are sequences of images. If specified,
            the score will be calculated separately for each image in the sequence. If ``None``, the score will be
            calculated across all dimensions.

    """
    _check_same_shape(preds, target)

    if keep_sequence_dim is None:
        sum_dims = None
    elif not 0 <= keep_sequence_dim < preds.ndim:
        raise ValueError(f"Expected keep_sequence dim to be in range [0, {preds.ndim}] but got {keep_sequence_dim}")
    else:
        sum_dims = tuple(i for i in range(preds.ndim) if i != keep_sequence_dim)

    # binarize the tensors with the threshold
    preds_bin = (preds >= threshold).bool()
    target_bin = (target >= threshold).bool()

    if keep_sequence_dim is None:
        hits = torch.sum(preds_bin & target_bin).int()
        misses = torch.sum((preds_bin ^ target_bin) & target_bin).int()
        false_alarms = torch.sum((preds_bin ^ target_bin) & preds_bin).int()
    else:
        hits = torch.sum(preds_bin & target_bin, dim=sum_dims).int()
        misses = torch.sum((preds_bin ^ target_bin) & target_bin, dim=sum_dims).int()
        false_alarms = torch.sum((preds_bin ^ target_bin) & preds_bin, dim=sum_dims).int()
    return hits, misses, false_alarms


def _critical_success_index_compute(hits: Tensor, misses: Tensor, false_alarms: Tensor) -> Tensor:
    """Compute critical success index.

    Args:
        hits: Number of true positives after binarization
        misses: Number of false negatives after binarization
        false_alarms: Number of false positives after binarization

    Returns:
        If input tensors are 5-dimensional and ``keep_sequence_dim=True``, the metric returns a ``(S,)`` vector
        with CSI scores for each image in the sequence. Otherwise, it returns a scalar tensor with the CSI score.

    """
    return _safe_divide(hits, hits + misses + false_alarms)


def critical_success_index(
    preds: Tensor, target: Tensor, threshold: float, keep_sequence_dim: Optional[int] = None
) -> Tensor:
    """Compute critical success index.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        threshold: Values above or equal to threshold are replaced with 1, below by 0
        keep_sequence_dim: Index of the sequence dimension if the inputs are sequences of images. If specified,
            the score will be calculated separately for each image in the sequence. If ``None``, the score will be
            calculated across all dimensions.

    Returns:
        If ``keep_sequence_dim`` is specified, the metric returns a vector of  with CSI scores for each image
        in the sequence. Otherwise, it returns a scalar tensor with the CSI score.

    Example:
        >>> import torch
        >>> from torchmetrics.functional.regression import critical_success_index
        >>> x = torch.Tensor([[0.2, 0.7], [0.9, 0.3]])
        >>> y = torch.Tensor([[0.4, 0.2], [0.8, 0.6]])
        >>> critical_success_index(x, y, 0.5)
        tensor(0.3333)

    Example:
        >>> import torch
        >>> from torchmetrics.functional.regression import critical_success_index
        >>> x = torch.Tensor([[[0.2, 0.7], [0.9, 0.3]], [[0.2, 0.7], [0.9, 0.3]]])
        >>> y = torch.Tensor([[[0.4, 0.2], [0.8, 0.6]], [[0.4, 0.2], [0.8, 0.6]]])
        >>> critical_success_index(x, y, 0.5, keep_sequence_dim=0)
        tensor([0.3333, 0.3333])

    """
    hits, misses, false_alarms = _critical_success_index_update(preds, target, threshold, keep_sequence_dim)
    return _critical_success_index_compute(hits, misses, false_alarms)
