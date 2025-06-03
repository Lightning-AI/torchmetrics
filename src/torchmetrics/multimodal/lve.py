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
from typing import Any, List, Optional, Sequence, Union

from torch import Tensor

from torchmetrics.functional.multimodal.lve import lip_vertex_error
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["LipVertexError.plot"]


class LipVertexError(Metric):
    r"""Implements Lip Vertex Error (LVE) metric for 3D talking head evaluation.

    The Lip Vertex Error (LVE) metric evaluates the quality of lip synchronization in 3D facial animations by measuring
    the maximum Euclidean distance (L2 error) between corresponding lip vertices of the generated and ground truth
    meshes for each frame. The metric is defined as:

    .. math::
        \text{LVE} = \frac{1}{N} \sum_{i=1}^{N} \max_{v \in \text{lip}} \|x_{i,v} - \hat{x}_{i,v}\|_2^2

    where :math:`N` is the number of frames, :math:`x_{i,v}` represents the 3D coordinates of vertex :math:`v` in the
    lip region of the ground truth frame :math:`i`, and :math:`\hat{x}_{i,v}` represents the corresponding vertex in the
    predicted frame. The metric computes the maximum squared L2 distance between corresponding lip vertices for each
    frame and averages across all frames. A lower LVE value indicates better lip synchronization quality.

    As input to ``forward`` and ``update``, the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): Predicted vertices tensor of shape (T, V, 3) where T is number of frames,
                V is number of vertices, and 3 represents XYZ coordinates
    - ``target`` (:class:`~torch.Tensor`): Ground truth vertices tensor of shape (T', V, 3) where T' can be different
                from T

    As output of ``forward`` and ``compute``, the metric returns the following output:

    - ``lve_score`` (:class:`~torch.Tensor`): A scalar tensor containing the mean Lip Vertex Error value across
                all frames.

    Args:
        mouth_map: List of vertex indices corresponding to the mouth region
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If the number of dimensions of `vertices_pred` or `vertices_gt` is not 3.
            If vertex dimensions (V) or coordinate dimensions (3) don't match
            If ``mouth_map`` is empty or contains invalid indices

    Example:
        >>> import torch
        >>> from torchmetrics.functional.multimodal import lip_vertex_error
        >>> vertices_pred = torch.randn(10, 100, 3, generator=torch.manual_seed(42))
        >>> vertices_gt = torch.randn(10, 100, 3, generator=torch.manual_seed(43))
        >>> mouth_map = [0, 1, 2, 3, 4]
        >>> lip_vertex_error(vertices_pred, vertices_gt, mouth_map)
        tensor(12.7688)

    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0

    vertices_pred_list: List[Tensor]
    vertices_gt_list: List[Tensor]

    def __init__(
        self,
        mouth_map: List[int],
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.mouth_map = mouth_map
        self.validate_args = validate_args

        if not self.mouth_map:
            raise ValueError("mouth_map cannot be empty.")

        self.add_state("vertices_pred_list", default=[], dist_reduce_fx=None)
        self.add_state("vertices_gt_list", default=[], dist_reduce_fx=None)

    def update(self, vertices_pred: Tensor, vertices_gt: Tensor) -> None:
        """Update metric states with predictions and targets.

        Args:
            vertices_pred: Predicted vertices tensor of shape (T, V, 3) where T is number of frames,
                V is number of vertices, and 3 represents XYZ coordinates
            vertices_gt: Ground truth vertices tensor of shape (T', V, 3) where T' can be different from T

        """
        if self.validate_args:
            if vertices_pred.ndim != 3 or vertices_gt.ndim != 3:
                raise ValueError(
                    f"Expected both vertices_pred and vertices_gt to have 3 dimensions but got "
                    f"{vertices_pred.ndim} and {vertices_gt.ndim} dimensions respectively."
                )
            if vertices_pred.shape[1:] != vertices_gt.shape[1:]:
                raise ValueError(
                    f"Expected vertices_pred and vertices_gt to have same vertex and coordinate dimensions but got "
                    f"shapes {vertices_pred.shape} and {vertices_gt.shape}."
                )
            if max(self.mouth_map) >= vertices_pred.shape[1]:
                raise ValueError(
                    f"mouth_map contains invalid vertex indices. Max index {max(self.mouth_map)} is larger than "
                    f"number of vertices {vertices_pred.shape[1]}."
                )

        min_frames = min(vertices_pred.shape[0], vertices_gt.shape[0])
        vertices_pred = vertices_pred[:min_frames]
        vertices_gt = vertices_gt[:min_frames]

        self.vertices_pred_list.append(vertices_pred)
        self.vertices_gt_list.append(vertices_gt)

    def compute(self) -> Tensor:
        """Compute the Lip Vertex Error over all accumulated states.

        Returns:
            torch.Tensor: A scalar tensor with the mean LVE value

        """
        vertices_pred = dim_zero_cat(self.vertices_pred_list)
        vertices_gt = dim_zero_cat(self.vertices_gt_list)
        return lip_vertex_error(vertices_pred, vertices_gt, self.mouth_map, self.validate_args)

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> import torch
            >>> from torchmetrics.multimodal.lve import LipVertexError
            >>> metric = LipVertexError(mouth_map=[0, 1, 2, 3, 4])
            >>> vertices_pred = torch.randn(10, 100, 3, generator=torch.manual_seed(42))
            >>> vertices_gt = torch.randn(10, 100, 3, generator=torch.manual_seed(43))
            >>> metric.update(vertices_pred, vertices_gt)
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.multimodal.lve import LipVertexError
            >>> metric = LipVertexError(mouth_map=[0, 1, 2, 3, 4])
            >>> values = []
            >>> for _ in range(10):
            ...     vertices_pred = torch.randn(10, 100, 3, generator=torch.manual_seed(42+_))
            ...     vertices_gt = torch.randn(10, 100, 3, generator=torch.manual_seed(43+_))
            ...     values.append(metric(vertices_pred, vertices_gt))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)
