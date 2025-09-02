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

from torchmetrics.functional.multimodal.fdd import upper_face_dynamics_deviation
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["UpperFaceDynamicsDeviation.plot"]


class UpperFaceDynamicsDeviation(Metric):
    r"""Implements the Upper Facial Dynamics Deviation (FDD) metric for 3D talking head evaluation.

    The FDD metric evaluates the quality of facial dynamics in 3D facial animations by measuring the deviation
    in the motion magnitude of upper-face vertices between the generated and ground truth meshes. It quantifies
    how well the predicted motion dynamics match the ground truth over time.

    The metric is defined as:

    .. math::
        \text{FDD} = \frac{1}{|\text{SU}|} \sum_{v \in \text{SU}} \Big( \text{std}(\|x_{1:T,v} - x_{0:T-1,v}\|_2) -
        \text{std}(\|\hat{x}_{1:T,v} - \hat{x}_{0:T-1,v}\|_2) \Big)

    where :math:`\text{SU}` is the set of upper-face vertex indices, 
    :math:`x_{t,v}` denotes the 3D coordinates of vertex :math:`v` in frame :math:`t` of the ground truth mesh, 
    :math:`\hat{x}_{t,v}` denotes the corresponding vertex in the predicted mesh, and :math:`\text{std}` is the
    standard deviation along the temporal axis.  

    The metric computes the standard deviation of the frame-to-frame L2 displacements of each upper-face vertex 
    for both predicted and ground truth sequences, then averages the differences over all upper-face vertices. 
    A lower FDD value indicates better temporal consistency of facial motion.

    As input to ``forward`` and ``update``, the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): Predicted vertices tensor of shape (T, V, 3) where T is the number of frames,
      V is the number of vertices, and 3 represents XYZ coordinates.
    - ``target`` (:class:`~torch.Tensor`): Ground truth vertices tensor of shape (T, V, 3) where T is the number of frames,
      V is the number of vertices, and 3 represents XYZ coordinates.
    - ``upper_face_map`` (:class:`list`): List of vertex indices corresponding to the upper-face region.

    As output of ``forward`` and ``compute``, the metric returns the following output:

    - ``fdd_score`` (:class:`~torch.Tensor`): A scalar tensor containing the mean Face Dynamics Deviation
      across all upper-face vertices.

    Args:
        upper_face_map: List of vertex indices for the upper-face region.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If the number of dimensions of `preds` or `target` is not 3.
            If vertex dimensions (V) or coordinate dimensions (3) do not match.
            If ``upper_face_map`` is empty or contains invalid indices.

    Example:
        >>> import torch
        >>> from torchmetrics.multimodal.fdd import UpperFaceDynamicsDeviation
        >>> metric = UpperFaceDynamicsDeviation(upper_face_map=[0, 1, 2, 3, 4])
        >>> vertices_pred = torch.randn(10, 100, 3, generator=torch.manual_seed(42))
        >>> vertices_gt = torch.randn(10, 100, 3, generator=torch.manual_seed(43))
        >>> metric(vertices_pred, vertices_gt)
        tensor(-0.3668)
    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0

    vertices_pred_list: List[Tensor]
    vertices_gt_list: List[Tensor]

    def __init__(
        self,
        upper_face_map: List[int],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.upper_face_map = upper_face_map

        if not self.upper_face_map:
            raise ValueError("upper_face_map cannot be empty.")

        self.add_state("vertices_pred_list", default=[], dist_reduce_fx=None)
        self.add_state("vertices_gt_list", default=[], dist_reduce_fx=None)

    def update(self, vertices_pred: Tensor, vertices_gt: Tensor) -> None:
        """Update metric states with predictions and targets.

        Args:
            vertices_pred: Predicted vertices tensor of shape (T, V, 3) where T is number of frames,
                V is number of vertices, and 3 represents XYZ coordinates
            vertices_gt: Ground truth vertices tensor of shape (T', V, 3) where T is number of frames,
                V is number of vertices, and 3 represents XYZ coordinates

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
        if max(self.upper_face_map) >= vertices_pred.shape[1]:
            raise ValueError(
                f"upper_face_map contains invalid vertex indices. Max index {max(self.upper_face_map)} is larger than "
                f"number of vertices {vertices_pred.shape[1]}."
            )

        self.vertices_pred_list.append(vertices_pred)
        self.vertices_gt_list.append(vertices_gt)

    def compute(self) -> Tensor:
        """Compute the Upper Face Dynamics Deviation over all accumulated states.

        Returns:
            torch.Tensor: A scalar tensor with the mean FDD value

        """
        vertices_pred = dim_zero_cat(self.vertices_pred_list)
        vertices_gt = dim_zero_cat(self.vertices_gt_list)
        return upper_face_dynamics_deviation(vertices_pred, vertices_gt, self.upper_face_map)

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
            >>> from torchmetrics.multimodal.fdd import UpperFaceDynamicsDeviation
            >>> metric = UpperFaceDynamicsDeviation(upper_face_map=[0, 1, 2, 3, 4])
            >>> vertices_pred = torch.randn(10, 100, 3, generator=torch.manual_seed(42))
            >>> vertices_gt = torch.randn(10, 100, 3, generator=torch.manual_seed(43))
            >>> metric.update(vertices_pred, vertices_gt)
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.multimodal.fdd import UpperFaceDynamicsDeviation
            >>> metric = UpperFaceDynamicsDeviation(upper_face_map=[0, 1, 2, 3, 4])
            >>> values = []
            >>> for _ in range(10):
            ...     vertices_pred = torch.randn(10, 100, 3, generator=torch.manual_seed(42+_))
            ...     vertices_gt = torch.randn(10, 100, 3, generator=torch.manual_seed(43+_))
            ...     values.append(metric(vertices_pred, vertices_gt))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)
