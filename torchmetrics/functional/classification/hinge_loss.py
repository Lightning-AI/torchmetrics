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

from torchmetrics.utilities.checks import _input_format_classification
from torchmetrics.utilities.enums import DataType


def hinge_loss(
        preds: Tensor,
        target: Tensor,
):
    _, target, mode = _input_format_classification(torch.ones_like(preds) / preds.shape[1], target)
    target = target.bool()

    if mode == DataType.MULTICLASS:
        margin = preds[target]
        margin -= torch.max(preds[~target].view(preds.shape[0], -1), dim=1)[0]

    losses = 1 - margin
    losses = torch.clamp(losses, 0)
    return torch.mean(losses)
