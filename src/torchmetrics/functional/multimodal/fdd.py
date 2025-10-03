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
    template: Tensor,
    upper_face_map: List[int],
) -> Tensor:
    r"""Compute Upper Face Dynamics Deviation (FDD) for 3D talking head evaluation.

    The Upper Face Dynamics Deviation (FDD) metric evaluates the quality of facial expressions in the upper
    face region for 3D talking head models. It quantifies the deviation in vertex motion dynamics between the
    predicted and ground truth sequences by comparing the temporal variation (standard deviation) of per-vertex
    squared displacements relative to a neutral template. Lower values of FDD indicate closer alignment of the
    predicted upper-face motion dynamics with the ground truth.

    The metric is defined as:

    .. math::
        \text{FDD} = \frac{1}{|S_U|} \sum_{v \in S_U} \Big( \text{std}(\| x_{1:T,v} -
        \text{template}_v \|_2^2) - \text{std}(\| \hat{x}_{1:T,v} - \text{template}_v \|_2^2) \Big)

    where :math:`T` is the number of frames, :math:`S_U` is the set of upper-face vertices with :math:`M = |S_U|`,
    :math:`x_{t,v}` are the 3D coordinates of vertex :math:`v` at frame :math:`t` in the ground truth sequence,
    and :math:`\hat{x}_{t,v} \in \mathbb{R}^3` are the corresponding predicted vertices. The neutral template coordinate
    of vertex :math:`v` is denoted as :math:`\text{template}_v \in \mathbb{R}^3`. The operator :math:`\text{std}(\cdot)`
    computes the standard deviation of the temporal sequence.

    Args:
        vertices_pred: Predicted vertices tensor of shape (T, V, 3) where T is number of frames,
            V is number of vertices, and 3 represents XYZ coordinates.
        vertices_gt: Ground truth vertices tensor of shape (T, V, 3) where T is number of frames,
            V is number of vertices, and 3 represents XYZ coordinates.
        template: Template mesh tensor of shape (V, 3) representing the neutral face.
        upper_face_map: List of vertex indices corresponding to the upper face region.

    Returns:
        torch.Tensor: Scalar tensor containing the mean FDD value across upper-face vertices.

    Raises:
        ValueError:
            If the number of dimensions of `vertices_pred` or `vertices_gt` is not 3.
            If vertex dimensions (V) or coordinate dimensions (3) don't match.
            If ``upper_face_map`` is empty or contains invalid indices.

    Example:
        >>> import torch
        >>> from torchmetrics.functional.multimodal import upper_face_dynamics_deviation
        >>> vertices_pred = torch.randn(10, 100, 3, generator=torch.manual_seed(41))
        >>> vertices_gt = torch.randn(10, 100, 3, generator=torch.manual_seed(42))
        >>> upper_face_map = [10, 11, 12, 13, 14]
        >>> template = torch.randn(100, 3, generator=torch.manual_seed(43))
        >>> upper_face_dynamics_deviation(vertices_pred, vertices_gt, template, upper_face_map)
        tensor(1.0385)

    """
    if vertices_pred.ndim != 3 or vertices_gt.ndim != 3:
        raise ValueError(
            f"Expected both vertices_pred and vertices_gt to have 3 dimensions but got "
            f"{vertices_pred.ndim} and {vertices_gt.ndim} dimensions respectively."
        )
    if template.ndim != 2 or template.shape[1] != 3:
        raise ValueError(f"Expected template to have shape (V, 3) but got {template.shape}.")
    if vertices_pred.shape[1:] != vertices_gt.shape[1:]:
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
    min_frames = min(vertices_pred.shape[0], vertices_gt.shape[0])
    pred = vertices_pred[:min_frames, upper_face_map, :]  # (T, M, 3)
    gt = vertices_gt[:min_frames, upper_face_map, :]
    template = template.to(pred.device)[upper_face_map, :]  # (M, 3)

    pred_disp = pred - template  # (T, M, 3)
    gt_disp = gt - template

    pred_norm_sq = torch.sum(pred_disp**2, dim=-1)  # (T, M)
    gt_norm_sq = torch.sum(gt_disp**2, dim=-1)  # (T, M)

    pred_dyn = torch.std(pred_norm_sq, dim=0, unbiased=False)  # (M,)
    gt_dyn = torch.std(gt_norm_sq, dim=0, unbiased=False)

    return torch.mean(gt_dyn - pred_dyn)  # scalar
