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
import sys
from collections.abc import Sequence
from typing import Any, List, Optional, Union

import torch
from lightning_utilities import apply_to_collection
from torch import Tensor

from torchmetrics.utilities.exceptions import TorchMetricsUserWarning
from torchmetrics.utilities.imports import _TORCH_LESS_THAN_2_6, _XLA_AVAILABLE
from torchmetrics.utilities.prints import rank_zero_warn

METRIC_EPS = 1e-6


def dim_zero_cat(x: Union[Tensor, List[Tensor]]) -> Tensor:
    """Concatenation along the zero dimension."""
    if isinstance(x, torch.Tensor):
        return x
    x = [y.unsqueeze(0) if y.numel() == 1 and y.ndim == 0 else y for y in x]
    if not x:  # empty list
        raise ValueError("No samples to concatenate")
    return torch.cat(x, dim=0)


def dim_zero_sum(x: Tensor) -> Tensor:
    """Summation along the zero dimension."""
    return torch.sum(x, dim=0)


def dim_zero_mean(x: Tensor) -> Tensor:
    """Average along the zero dimension."""
    return torch.mean(x, dim=0)


def dim_zero_max(x: Tensor) -> Tensor:
    """Max along the zero dimension."""
    return torch.max(x, dim=0).values


def dim_zero_min(x: Tensor) -> Tensor:
    """Min along the zero dimension."""
    return torch.min(x, dim=0).values


def _flatten(x: Sequence) -> list:
    """Flatten list of list into single list."""
    return [item for sublist in x for item in sublist]


def _flatten_dict(x: dict) -> tuple[dict, bool]:
    """Flatten dict of dicts into single dict and checking for duplicates in keys along the way."""
    new_dict = {}
    duplicates = False
    for key, value in x.items():
        if isinstance(value, dict):
            for k, v in value.items():
                if k in new_dict:
                    duplicates = True
                new_dict[k] = v
        else:
            if key in new_dict:
                duplicates = True
            new_dict[key] = value
    return new_dict, duplicates


def to_onehot(
    label_tensor: Tensor,
    num_classes: Optional[int] = None,
) -> Tensor:
    """Convert  a dense label tensor to one-hot format.

    Args:
        label_tensor: dense label tensor, with shape [N, d1, d2, ...]
        num_classes: number of classes C

    Returns:
        A sparse label tensor with shape [N, C, d1, d2, ...]

    Example:
        >>> x = torch.tensor([1, 2, 3])
        >>> to_onehot(x)
        tensor([[0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])

    """
    if num_classes is None:
        num_classes = int(label_tensor.max().detach().item() + 1)

    tensor_onehot = torch.zeros(
        label_tensor.shape[0],
        num_classes,
        *label_tensor.shape[1:],
        dtype=label_tensor.dtype,
        device=label_tensor.device,
    )
    index = label_tensor.long().unsqueeze(1).expand_as(tensor_onehot)
    return tensor_onehot.scatter_(1, index, 1.0)


def _top_k_with_half_precision_support(x: Tensor, k: int = 1, dim: int = 1) -> Tensor:
    """torch.top_k does not support half precision on CPU."""
    if x.dtype == torch.half and not x.is_cuda:
        idx = torch.argsort(x, dim=dim, stable=True).flip(dim)
        return idx.narrow(dim, 0, k)
    return x.topk(k=k, dim=dim).indices


def select_topk(prob_tensor: Tensor, topk: int = 1, dim: int = 1) -> Tensor:
    """Convert a probability tensor to binary by selecting top-k the highest entries.

    Args:
        prob_tensor: dense tensor of shape ``[..., C, ...]``, where ``C`` is in the
            position defined by the ``dim`` argument
        topk: number of the highest entries to turn into 1s
        dim: dimension on which to compare entries

    Returns:
        A binary tensor of the same shape as the input tensor of type ``torch.int32``

    Example:
        >>> x = torch.tensor([[1.1, 2.0, 3.0], [2.0, 1.0, 0.5]])
        >>> select_topk(x, topk=2)
        tensor([[0, 1, 1],
                [1, 1, 0]], dtype=torch.int32)

    """
    topk_tensor = torch.zeros_like(prob_tensor, dtype=torch.int)
    if topk == 1:  # argmax has better performance than topk
        topk_tensor.scatter_(dim, prob_tensor.argmax(dim=dim, keepdim=True), 1.0)
    else:
        topk_tensor.scatter_(dim, _top_k_with_half_precision_support(prob_tensor, k=topk, dim=dim), 1.0)
    return topk_tensor.int()


def to_categorical(x: Tensor, argmax_dim: int = 1) -> Tensor:
    """Convert  a tensor of probabilities to a dense label tensor.

    Args:
        x: probabilities to get the categorical label [N, d1, d2, ...]
        argmax_dim: dimension to apply

    Return:
        A tensor with categorical labels [N, d2, ...]

    Example:
        >>> x = torch.tensor([[0.2, 0.5], [0.9, 0.1]])
        >>> to_categorical(x)
        tensor([1, 0])

    """
    return torch.argmax(x, dim=argmax_dim)


def _squeeze_scalar_element_tensor(x: Tensor) -> Tensor:
    return x.squeeze() if x.numel() == 1 else x


def _squeeze_if_scalar(data: Any) -> Any:
    return apply_to_collection(data, Tensor, _squeeze_scalar_element_tensor)


def _bincount(x: Tensor, minlength: Optional[int] = None) -> Tensor:
    """Implement custom bincount.

    PyTorch currently does not support ``torch.bincount`` when running in deterministic mode on GPU or when running
    MPS devices or when running on XLA device. This implementation therefore falls back to using a combination of
    `torch.arange` and `torch.eq` in these scenarios. A small performance hit can expected and higher memory consumption
    as `[batch_size, mincount]` tensor needs to be initialized compared to native ``torch.bincount``.

    Args:
        x: tensor to count
        minlength: minimum length to count

    Returns:
        Number of occurrences for each unique element in x

    Example:
        >>> x = torch.tensor([0,0,0,1,1,2,2,2,2])
        >>> _bincount(x, minlength=3)
        tensor([3, 2, 4])

    """
    if minlength is None:
        minlength = len(torch.unique(x))

    if torch.are_deterministic_algorithms_enabled() or _XLA_AVAILABLE or x.is_mps:
        mesh = torch.arange(minlength, device=x.device).repeat(len(x), 1)
        return torch.eq(x.reshape(-1, 1), mesh).sum(dim=0)

    return torch.bincount(x, minlength=minlength)


def _cumsum(x: Tensor, dim: Optional[int] = 0, dtype: Optional[torch.dtype] = None) -> Tensor:
    """Implement custom cumulative summation for Torch versions which does not support it natively."""
    is_cuda_fp_deterministic = torch.are_deterministic_algorithms_enabled() and x.is_cuda and x.is_floating_point()
    if _TORCH_LESS_THAN_2_6 and is_cuda_fp_deterministic and sys.platform != "win32":
        rank_zero_warn(
            "You are trying to use a metric in deterministic mode on GPU that uses `torch.cumsum`, which is currently"
            " not supported. The tensor will be copied to the CPU memory to compute it and then copied back to GPU."
            " Expect some slowdowns.",
            TorchMetricsUserWarning,
        )
        return x.cpu().cumsum(dim=dim, dtype=dtype).to(x.device)
    return torch.cumsum(x, dim=dim, dtype=dtype)


def _flexible_bincount(x: Tensor) -> Tensor:
    """Similar to `_bincount`, but works also with tensor that do not contain continuous values.

    Args:
        x: tensor to count

    Returns:
        Number of occurrences for each unique element in x

    """
    # make sure elements in x start from 0
    x = x - x.min()
    unique_x = torch.unique(x)

    output = _bincount(x, minlength=torch.max(unique_x) + 1)  # type: ignore[arg-type]
    # remove zeros from output tensor
    return output[unique_x]


def allclose(tensor1: Tensor, tensor2: Tensor) -> bool:
    """Wrap torch.allclose to be robust towards dtype difference."""
    if tensor1.dtype != tensor2.dtype:
        tensor2 = tensor2.to(dtype=tensor1.dtype)
    return torch.allclose(tensor1, tensor2)


def interp(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    """Interpolation function comparable to numpy.interp.

    Args:
        x: x-coordinates where to evaluate the interpolated values
        xp: x-coordinates of the data points
        fp: y-coordinates of the data points

    """
    # Sort xp and fp based on xp for compatibility with np.interp
    sorted_indices = torch.argsort(xp)
    xp = xp[sorted_indices]
    fp = fp[sorted_indices]

    # Calculate slopes for each interval
    slopes = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])

    # Identify where x falls relative to xp
    indices = torch.searchsorted(xp, x) - 1
    indices = torch.clamp(indices, 0, len(slopes) - 1)

    # Compute interpolated values
    return fp[indices] + slopes[indices] * (x - xp[indices])
