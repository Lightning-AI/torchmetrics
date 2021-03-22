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
from typing import Tuple

import torch
from torch import Tensor, tensor

from torchmetrics.utilities.checks import _check_shape_and_type_consistency
from torchmetrics.utilities.data import to_onehot
from torchmetrics.utilities.enums import DataType


def _hinge_loss_update(preds: Tensor, target: Tensor, squared: bool = False) -> Tuple[Tensor, Tensor]:
    mode, implied_classes = _check_shape_and_type_consistency(preds, target)

    if mode == DataType.MULTICLASS:
        target = to_onehot(target, max(2, implied_classes)).bool()
        margin = preds[target]
        margin -= torch.max(preds[~target].view(preds.shape[0], -1), dim=1)[0]
    elif mode == DataType.BINARY:
        target = target.bool()
        margin = torch.zeros_like(preds)
        margin[target] = preds[target]
        margin[~target] = - preds[~target]
    else:
        raise ValueError(
            "Either `preds` and `target` both should have the (same) shape (N), or `target` should be (N) and `preds`"
            " should be (N, C)."
        )

    losses = 1 - margin
    losses = torch.clamp(losses, 0)

    if squared:
        losses = losses.pow(2)

    total = tensor(target.shape[0], device=target.device)
    return losses.sum(), total


def _hinge_loss_compute(loss: Tensor, total: Tensor) -> Tensor:
    return loss / total


def hinge_loss(
        preds: Tensor,
        target: Tensor,
        squared: bool = False
):
    loss, total = _hinge_loss_update(preds, target, squared=squared)
    return _hinge_loss_compute(loss, total)
