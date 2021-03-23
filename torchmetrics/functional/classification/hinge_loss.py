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
from typing import Tuple, Optional

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
):
    loss, total = _hinge_loss_update(preds, target, squared=squared, multiclass_mode=multiclass_mode)
    return _hinge_loss_compute(loss, total)
