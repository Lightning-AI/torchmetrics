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
import torch
from torch import Tensor

ALLOWED_BOOL_TYPES = (torch.bool,)
ALLOWED_INT_TYPES = (torch.int8, torch.int16, torch.int32, torch.int64)
ALLOWED_FLOAT_TYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def retrieval_average_precision(preds: Tensor, target: Tensor) -> Tensor:
    r"""
    Computes average precision (for information retrieval), as explained
    `here <https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision>`__.

    `preds` and `target` should be of the same shape and live on the same device. If no `target` is ``True``,
    `0` is returned. Target must be of type `bool` or `int`, otherwise an error is raised.

    Args:
        preds: estimated probabilities of each document to be relevant.
        target: ground truth about each document being relevant or not. Requires `bool` or `int` tensor.

    Return:
        a single-value tensor with the average precision (AP) of the predictions `preds` wrt the labels `target`.

    Example:
        >>> preds = torch.tensor([0.2, 0.3, 0.5])
        >>> target = torch.tensor([True, False, True])
        >>> retrieval_average_precision(preds, target)
        tensor(0.8333)
    """

    if preds.shape != target.shape or preds.device != target.device:
        raise ValueError("`preds` and `target` must have the same shape and live on the same device")

    if target.numel() == 0 or preds.numel() == 0:
        raise ValueError("`target` and `preds` must be non-empty tensors")

    if preds.dtype not in ALLOWED_INT_TYPES + ALLOWED_FLOAT_TYPES:
        raise ValueError("`target` must be a tensor of floats or integers")

    if target.dtype not in ALLOWED_BOOL_TYPES + ALLOWED_INT_TYPES:
        raise ValueError("`target` must be a tensor of booleans or integers")

    if target.sum() == 0:
        return torch.tensor(0.0, device=preds.device)

    if target.dtype is not torch.bool:
        target = target.bool()

    target = target[torch.argsort(preds, dim=-1, descending=True)]
    positions = torch.arange(1, len(target) + 1, device=target.device, dtype=torch.float32)[target > 0]
    res = torch.div((torch.arange(len(positions), device=positions.device, dtype=torch.float32) + 1), positions).mean()
    return res
