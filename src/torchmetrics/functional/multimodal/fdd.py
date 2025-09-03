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
from typing import List

import torch
from torch import Tensor


def upper_face_dynamics_deviation(
    vertices_pred: Tensor,
    vertices_gt: Tensor,
    upper_face_map: List[int],
) -> Tensor:
    r"""Compute Upper Face Dynamics Deviation (FDD) for 3D talking head evaluation.

    The Upper Face Dynamics Deviation (FDD) metric evaluates the quality of facial expressions in the upper
    face region for 3D talking head models. It quantifies the deviation in vertex motion dynamics between the
    predicted and ground truth sequences by comparing the frame-to-frame displacement of vertices.

    The metric is defined as:

    .. math::
        \text{FDD} = \frac{1}{N-1} \sum_{i=1}^{N-1} \frac{1}{M} \sum_{v \in \text{upper}}
        \big\| (x_{i+1,v} - x_{i,v}) - (\hat{x}_{i+1,v} - \hat{x}_{i,v}) \big\|_2^2

    where :math:`N` is the number of frames, :math:`M` is the number of vertices in the upper face region,
    :math:`x_{i,v}` are the 3D coordinates of vertex :math:`v` at frame :math:`i` in the ground truth sequence,
    and :math:`\hat{x}_{i,v}` are the corresponding predicted vertices. The metric measures the mean squared L2
    deviation of inter-frame motion dynamics. Lower values indicate closer alignment of facial dynamics.

    Args:
        vertices_pred: Predicted vertices tensor of shape (T, V, 3) where T is number of frames,
            V is number of vertices, and 3 represents XYZ coordinates.
        vertices_gt: Ground truth vertices tensor of shape (T, V, 3) where T is number of frames,
            V is number of vertices, and 3 represents XYZ coordinates.
        upper_face_map: List of vertex indices corresponding to the upper face region.

    Returns:
        torch.Tensor: Scalar tensor containing the mean FDD value across all frames.

    Raises:
        ValueError:
            If the number of dimensions of `vertices_pred` or `vertices_gt` is not 3.
            If vertex dimensions (V) or coordinate dimensions (3) don't match.
            If ``upper_face_map`` is empty or contains invalid indices.
            If there are at least two frames to compute face dynamics deviation.

    Example:
        >>> import torch
        >>> from torchmetrics.functional.multimodal import upper_face_dynamics_deviation
        >>> vertices_pred = torch.randn(10, 100, 3, generator=torch.manual_seed(42))
        >>> vertices_gt = torch.randn(10, 100, 3, generator=torch.manual_seed(43))
        >>> upper_face_map = [10, 11, 12, 13, 14]
        >>> upper_face_dynamics_deviation(vertices_pred, vertices_gt, upper_face_map)
        tensor(0.1176)

    """
    if vertices_pred.ndim != 3 or vertices_gt.ndim != 3:
        raise ValueError(
            f"Expected both vertices_pred and vertices_gt to have 3 dimensions but got "
            f"{vertices_pred.ndim} and {vertices_gt.ndim} dimensions respectively."
        )
    if vertices_pred.shape != vertices_gt.shape:
        raise ValueError(
            f"Expected vertices_pred and vertices_gt to have same vertex and coordinate dimensions but got "
            f"shapes {vertices_pred.shape} and {vertices_gt.shape}."
        )
    if not upper_face_map:
        raise ValueError("upper_face_map cannot be empty.")
    if max(upper_face_map) >= vertices_pred.shape[1]:
        raise ValueError(
            f"upper_face_map contains invalid vertex indices. Max index {max(upper_face_map)} is larger than "
            f"number of vertices {vertices_pred.shape[1]}."
        )
    if vertices_pred.shape[0] < 2:
        raise ValueError("Need at least 2 frames to compute dynamics deviation.")

    pred = vertices_pred[:, upper_face_map, :]  # (T, M, 3)
    gt = vertices_gt[:, upper_face_map, :]

    pred_disp = pred[1:] - pred[:-1]  # (T-1, M, 3)
    gt_disp = gt[1:] - gt[:-1]

    pred_norm = torch.linalg.norm(pred_disp, dim=-1)  # (T-1, M)
    gt_norm = torch.linalg.norm(gt_disp, dim=-1)

    pred_dyn = torch.std(pred_norm, dim=0, unbiased=False)  # (M,)
    gt_dyn = torch.std(gt_norm, dim=0, unbiased=False)

    return torch.mean(gt_dyn - pred_dyn)  # scalar
