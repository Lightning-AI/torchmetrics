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


def _safe_matmul(x: Tensor, y: Tensor) -> Tensor:
    """Safe calculation of matrix multiplication.

    If input is float16, will cast to float32 for computation and back again.
    """
    if x.dtype == torch.float16 or y.dtype == torch.float16:
        return (x.float() @ y.T.float()).half()
    return x @ y.T


def _safe_xlogy(x: Tensor, y: Tensor) -> Tensor:
    """Computes x * log(y). Returns 0 if x=0.

    Example:
        >>> import torch
        >>> x = torch.zeros(1)
        >>> _safe_xlogy(x, 1/x)
        tensor([0.])

    """
    res = x * torch.log(y)
    res[x == 0] = 0.0
    return res


# From https://github.com/pytorch/pytorch/issues/21987#issuecomment-539402619
# From PyTorch v1.10, this is officially supported.
def nanmean(v: Tensor, *args, inplace: bool = False, **kwargs) -> Tensor:  # type: ignore
    """Computes the mean of all non-NaN elements along the specified dimensions.
    
    Args:
        v: input tensor
        inplace: if operation should done inplace
        *args, **kwargs: additional arguments like ``dim`` that should be passed to the mean
            operator
    """
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)
