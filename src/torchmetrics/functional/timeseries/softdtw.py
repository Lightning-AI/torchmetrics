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
import math
from typing import Callable, Optional

import torch
from torch import Tensor


def _soft_dtw_validate_args(preds: Tensor, target: Tensor, gamma: float) -> None:
    """Validate the input arguments for the soft_dtw function."""
    if preds.ndim != 3 or target.ndim != 3:
        raise ValueError("Inputs preds and target must be 3-dimensional tensors of shape [B, N, D] and [B, M, D].")
    if preds.shape[0] != target.shape[0]:
        raise ValueError("Batch size of preds and target must be the same.")
    if preds.shape[2] != target.shape[2]:
        raise ValueError("Feature dimension of preds and target must be the same.")
    if not isinstance(gamma, float) or gamma <= 0:
        raise ValueError("Gamma must be a positive float.")


def _soft_dtw_compute(preds: Tensor, target: Tensor, gamma: float, distance_fn: Optional[Callable] = None) -> Tensor:
    """Compute the Soft-DTW distance between two batched sequences."""
    B, N, D = preds.shape
    _, M, _ = target.shape
    device, dtype = target.device, target.dtype
    if preds.dtype != target.dtype:
        target = target.to(preds.dtype)

    if distance_fn is None:

        def distance_fn(a, b):
            return torch.cdist(a, b, p=2).pow(2)

    D = distance_fn(preds, target)  # [B, N, M]

    R = torch.ones((B, N + 2, M + 2), device=device, dtype=dtype) * math.inf
    R[:, 0, 0] = 0.0

    def softmin(a, b, c, gamma):
        vals = torch.stack([a, b, c], dim=-1)
        return -gamma * torch.logsumexp(-vals / gamma, dim=-1)

    # Loop based implementation
    # for i in range(1, N + 1):
    #     for j in range(1, M + 1):
    #         r1 = R[:, i-1, j-1]
    #         r2 = R[:, i-1, j]
    #         r3 = R[:, i, j-1]
    #         R[:, i, j] = D[:, i-1, j-1] + softmin(r1, r2, r3, gamma)

    # Anti-diagonal implementation
    for k in range(2, N + M + 1):
        i_vals = torch.arange(1, N + 1, device=device)
        j_vals = k - i_vals
        mask = (j_vals >= 1) & (j_vals <= M)
        i_vals = i_vals[mask]
        j_vals = j_vals[mask]

        if len(i_vals) == 0:
            continue

        r1 = R[:, i_vals - 1, j_vals - 1]
        r2 = R[:, i_vals - 1, j_vals]
        r3 = R[:, i_vals, j_vals - 1]
        R[:, i_vals, j_vals] = D[:, i_vals - 1, j_vals - 1] + softmin(r1, r2, r3, gamma)

    return R[:, N, M]


def soft_dtw(
    preds: Tensor,
    target: Tensor,
    gamma: float = 1.0,
    distance_fn=None,
) -> Tensor:
    r"""Compute the **Soft Dynamic Time Warping (Soft-DTW)** distance between two batched sequences.

    This is a differentiable relaxation of the classic Dynamic Time Warping (DTW) algorithm, introduced by
    Marco Cuturi and Mathieu Blondel (2017).
    It replaces the hard minimum in DTW recursion with a *soft-minimum* using a log-sum-exp formulation:

    .. math::
        \text{softmin}_\gamma(a,b,c) = -\gamma \log \left( e^{-a/\gamma} + e^{-b/\gamma} + e^{-c/\gamma} \right)

    The Soft-DTW recurrence is then defined as:

    .. math::
        R_{i,j} = D_{i,j} + \text{softmin}_\gamma(R_{i-1,j}, R_{i,j-1}, R_{i-1,j-1})

    where :math:`D_{i,j}` is the pairwise distance between sequence elements :math:`x_i` and :math:`y_j`.

    The final Soft-DTW distance is :math:`R_{N,M}`.

    Args:
        x: Tensor of shape ``[B, N, D]`` — batch of input sequences.
        y: Tensor of shape ``[B, M, D]`` — batch of target sequences.
        gamma: Smoothing parameter (:math:`\gamma > 0`). Smaller values make the loss closer to standard DTW (hard minimum),
            while larger values produce a smoother and more differentiable surface.
        distance_fn: Optional callable ``(x, y) -> [B, N, M]`` defining the pairwise distance matrix.
            If ``None``, defaults to **squared Euclidean distance**.

    Returns:
        A tensor of shape ``[B]`` containing the Soft-DTW distance for each sequence pair in the batch.

    Example::
        >>> import torch
        >>> from torchmetrics.functional.timeseries import soft_dtw
        >>>
        >>> x = torch.tensor([[[0.0], [1.0], [2.0]]])  # [B, N, D]
        >>> y = torch.tensor([[[0.0], [2.0], [3.0]]])  # [B, M, D]
        >>> soft_dtw(x, y, gamma=0.1)
        tensor([0.4003])

    Example (custom distance function)::
        >>> def cosine_dist(a, b):
        ...     a = torch.nn.functional.normalize(a, dim=-1)
        ...     b = torch.nn.functional.normalize(b, dim=-1)
        ...     return 1 - torch.bmm(a, b.transpose(1, 2))
        >>>
        >>> x = torch.randn(2, 5, 3)
        >>> y = torch.randn(2, 6, 3)
        >>> soft_dtw(x, y, gamma=0.5, distance_fn=cosine_dist)
        tensor([2.8301, 3.0128])

    """
    _soft_dtw_validate_args(preds, target, gamma)
    return _soft_dtw_compute(preds, target, gamma, distance_fn)
