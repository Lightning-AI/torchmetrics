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
from typing import Optional, Tuple

import torch
from torch import Tensor, tensor

from torchmetrics.utilities.data import to_onehot
from torchmetrics.utilities.enums import DataType


def _check_shape_and_type_consistency_hinge_loss(
        preds: Tensor,
        target: Tensor,
) -> DataType:
    if target.ndim > 1:
        raise ValueError(
            f"The `target` should be one dimensional, got `target` with shape={target.shape}.",
        )

    if preds.ndim == 1:
        if preds.shape != target.shape:
            raise ValueError(
                "The `preds` and `target` should have the same shape,",
                f" got `preds` with shape={preds.shape} and `target` with shape={target.shape}.",
            )
        mode = DataType.BINARY
    elif preds.ndim == 2:
        if preds.shape[0] != target.shape[0]:
            raise ValueError(
                "The `preds` and `target` should have the same shape in the first dimension,",
                f" got `preds` with shape={preds.shape} and `target` with shape={target.shape}.",
            )
        mode = DataType.MULTICLASS
    else:
        raise ValueError(
            f"The `preds` should be one or two dimensional, got `preds` with shape={preds.shape}."
        )
    return mode


def _hinge_loss_update(
        preds: Tensor,
        target: Tensor,
        squared: bool = False,
        multiclass_mode: Optional[str] = None,
) -> Tuple[Tensor, Tensor]:
    if preds.shape[0] == 1:
        preds, target = preds.squeeze().unsqueeze(0), target.squeeze().unsqueeze(0)
    else:
        preds, target = preds.squeeze(), target.squeeze()

    mode = _check_shape_and_type_consistency_hinge_loss(preds, target)

    if mode == DataType.MULTICLASS:
        target = to_onehot(target, max(2, preds.shape[1])).bool()

    if mode == DataType.MULTICLASS and (multiclass_mode is None or multiclass_mode == 'crammer_singer'):
        margin = preds[target]
        margin -= torch.max(preds[~target].view(preds.shape[0], -1), dim=1)[0]
    elif mode == DataType.BINARY or multiclass_mode == 'one_vs_all':
        target = target.bool()
        margin = torch.zeros_like(preds)
        margin[target] = preds[target]
        margin[~target] = - preds[~target]
    else:
        raise ValueError(
            "The `multiclass_mode` should be either None / 'crammer_singer' (default) or 'one_vs_all',"
            f" got {multiclass_mode}."
        )

    losses = 1 - margin
    losses = torch.clamp(losses, 0)

    if squared:
        losses = losses.pow(2)

    total = tensor(target.shape[0], device=target.device)
    return losses.sum(dim=0), total


def _hinge_loss_compute(loss: Tensor, total: Tensor) -> Tensor:
    return loss / total


def hinge_loss(
        preds: Tensor,
        target: Tensor,
        squared: bool = False,
        multiclass_mode: Optional[str] = None,
) -> Tensor:
    r"""
    Computes the mean `Hinge loss <https://en.wikipedia.org/wiki/Hinge_loss>`_, typically used for Support Vector
    Machines (SVMs). In the binary case it is defined as:

    .. math::
        \text{Hinge loss} = \max(0, 1 - y \times \hat{y})

    Where :math:`y \in {-1, 1}` is the target, and :math:`\hat{y} \in \mathbb{R}` is the prediction.

    In the multi-class case, when ``multiclass_mode=None`` (default) or ``multiclass_mode="crammer_singer"``, this
    metric will compute the multi-class hinge loss defined by Crammer and Singer as:

    .. math::
        \text{Hinge loss} = \max(0, 1 - \hat{y}_y + \max_{i \ne y} \hat{y}_i)

    Where :math:`y \in {0, ..., C}` is the target class, and :math:`\hat{y} \in \mathbb{R}^C` is the predicted output
    per class.

    In the multi-class case when ``multiclass_mode='one_vs_all'``, this metric will use a one-vs-all approach to compute
    the hinge loss, giving a vector of C outputs where each entry pits that class against all remaining classes.

    This metric can optionally output the mean of the squared hinge loss by setting ``squared=True``

    Only accepts inputs with preds shape of (N) (binary) or (N, C) (multi-class) and target shape of (N).

    Args:
        preds: Predictions from model (as float outputs from decision function).
        target: Ground truth labels.
        squared:
            If True, this will compute the squared hinge loss. Otherwise, computes the regular hinge loss (default).
        multiclass_mode:
            Which approach to use for multi-class inputs (has no effect in the binary case). ``None`` (default) or
            ``"crammer_singer"``, uses the Crammer Singer multi-class hinge loss. ``"one_vs_all"`` computes the hinge
            loss in a one-vs-all fashion.

    Raises:
        ValueError:
            If preds shape is not of size (N) or (N, C).
        ValueError:
            If target shape is not of size (N).
        ValueError:
            If ``multiclass_mode`` is not: None, ``"crammer_singer"``, or ``"one_vs_all"``.

    Example:
        >>> from torchmetrics.functional import hinge_loss
        >>> target = torch.tensor([0, 1, 1])
        >>> preds = torch.tensor([-2.2, 2.4, 0.1])
        >>> hinge_loss(preds, target)
        tensor(0.3000)

        >>> target = torch.tensor([0, 1, 2])
        >>> preds = torch.tensor([[-1.0, 0.9, 0.2], [0.5, -1.1, 0.8], [2.2, -0.5, 0.3]])
        >>> hinge_loss(preds, target)
        tensor(2.9000)
    """
    loss, total = _hinge_loss_update(preds, target, squared=squared, multiclass_mode=multiclass_mode)
    return _hinge_loss_compute(loss, total)
