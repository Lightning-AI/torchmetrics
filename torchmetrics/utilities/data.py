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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import torch
from torch import Tensor, tensor

from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_1_6, _TORCH_GREATER_EQUAL_1_7, _TORCH_GREATER_EQUAL_1_8

if _TORCH_GREATER_EQUAL_1_8:
    deterministic = torch.are_deterministic_algorithms_enabled
elif _TORCH_GREATER_EQUAL_1_7:
    deterministic = torch.is_deterministic
elif _TORCH_GREATER_EQUAL_1_6:
    deterministic = torch._is_deterministic
else:

    def deterministic() -> bool:
        return True


METRIC_EPS = 1e-6


def dim_zero_cat(x: Union[Tensor, List[Tensor]]) -> Tensor:
    """Concatenation along the zero dimension."""
    x = x if isinstance(x, (list, tuple)) else [x]
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


def _flatten_dict(x: Dict) -> Dict:
    """Flatten dict of dicts into single dict."""
    new_dict = {}
    for key, value in x.items():
        if isinstance(value, dict):
            for k, v in value.items():
                new_dict[k] = v
        else:
            new_dict[key] = value
    return new_dict


def to_onehot(
    label_tensor: Tensor,
    num_classes: Optional[int] = None,
) -> Tensor:
    """Converts a dense label tensor to one-hot format.

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
    zeros = torch.zeros_like(prob_tensor)
    if topk == 1:  # argmax has better performance than topk
        topk_tensor = zeros.scatter(dim, prob_tensor.argmax(dim=dim, keepdim=True), 1.0)
    else:
        topk_tensor = zeros.scatter(dim, prob_tensor.topk(k=topk, dim=dim).indices, 1.0)
    return topk_tensor.int()


def to_categorical(x: Tensor, argmax_dim: int = 1) -> Tensor:
    """Converts a tensor of probabilities to a dense label tensor.

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


def apply_to_collection(
    data: Any,
    dtype: Union[type, tuple],
    function: Callable,
    *args: Any,
    wrong_dtype: Optional[Union[type, tuple]] = None,
    **kwargs: Any,
) -> Any:
    """Recursively applies a function to all elements of a certain dtype.

    Args:
        data: the collection to apply the function to
        dtype: the given function will be applied to all elements of this dtype
        function: the function to apply
        *args: positional arguments (will be forwarded to call of ``function``)
        wrong_dtype: the given function won't be applied if this type is specified and the given collections is of
            the :attr:`wrong_type` even if it is of type :attr`dtype`
        **kwargs: keyword arguments (will be forwarded to call of ``function``)

    Returns:
        the resulting collection

    Example:
        >>> apply_to_collection(torch.tensor([8, 0, 2, 6, 7]), dtype=Tensor, function=lambda x: x ** 2)
        tensor([64,  0,  4, 36, 49])
        >>> apply_to_collection([8, 0, 2, 6, 7], dtype=int, function=lambda x: x ** 2)
        [64, 0, 4, 36, 49]
        >>> apply_to_collection(dict(abc=123), dtype=int, function=lambda x: x ** 2)
        {'abc': 15129}
    """
    elem_type = type(data)

    # Breaking condition
    if isinstance(data, dtype) and (wrong_dtype is None or not isinstance(data, wrong_dtype)):
        return function(data, *args, **kwargs)

    # Recursively apply to collection items
    if isinstance(data, Mapping):
        return elem_type({k: apply_to_collection(v, dtype, function, *args, **kwargs) for k, v in data.items()})

    if isinstance(data, tuple) and hasattr(data, "_fields"):  # named tuple
        return elem_type(*(apply_to_collection(d, dtype, function, *args, **kwargs) for d in data))

    if isinstance(data, Sequence) and not isinstance(data, str):
        return elem_type([apply_to_collection(d, dtype, function, *args, **kwargs) for d in data])

    # data is neither of dtype, nor a collection
    return data


def get_group_indexes(indexes: Tensor) -> List[Tensor]:
    """Given an integer ``indexes``, return indexes for each different value in ``indexes``.

    Args:
        indexes:

    Return:
        A list of integer ``torch.Tensor``s

    Example:
        >>> indexes = torch.tensor([0, 0, 0, 1, 1, 1, 1])
        >>> get_group_indexes(indexes)
        [tensor([0, 1, 2]), tensor([3, 4, 5, 6])]
    """

    res: dict = {}
    for i, _id in enumerate(indexes):
        _id = _id.item()
        if _id in res:
            res[_id] += [i]
        else:
            res[_id] = [i]

    return [tensor(x, dtype=torch.long) for x in res.values()]


def _squeeze_scalar_element_tensor(x: Tensor) -> Tensor:
    return x.squeeze() if x.numel() == 1 else x


def _squeeze_if_scalar(data: Any) -> Any:
    return apply_to_collection(data, Tensor, _squeeze_scalar_element_tensor)


def _bincount(x: Tensor, minlength: Optional[int] = None) -> Tensor:
    """``torch.bincount`` currently does not support deterministic mode on GPU.

    This implementation fallback to a for-loop counting occurrences in that case.

    Args:
        x: tensor to count
        minlength: minimum length to count

    Returns:
        Number of occurrences for each unique element in x
    """
    if x.is_cuda and deterministic():
        if minlength is None:
            minlength = len(torch.unique(x))
        output = torch.zeros(minlength, device=x.device, dtype=torch.long)
        for i in range(minlength):
            output[i] = (x == i).sum()
        return output
    else:
        return torch.bincount(x, minlength=minlength)


def allclose(tensor1: Tensor, tensor2: Tensor) -> bool:
    """Wrapper of torch.allclose that is robust towards dtype difference."""
    if tensor1.dtype != tensor2.dtype:
        tensor2 = tensor2.to(dtype=tensor1.dtype)
    return torch.allclose(tensor1, tensor2)
