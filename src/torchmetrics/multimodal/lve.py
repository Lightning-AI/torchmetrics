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

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from torchmetrics.functional.multimodal.lve import lip_vertex_error

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["LipVertexError.plot"]


class LipVertexError(Metric):
    """Implements Lip Vertex Error (LVE) metric for 3D talking head evaluation.

    The Lip Vertex Error (LVE) metric evaluates the quality of lip synchronization in 3D facial animations by measuring
    the maximum Euclidean distance (L2 error) between corresponding lip vertices of the generated and ground truth meshes
    for each frame.

    Args:
        mouth_map: List of vertex indices corresponding to the mouth region
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0

    def __init__(
        self,
        mouth_map: List[int],
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.mouth_map = mouth_map
        self.validate_args = validate_args

        # Initialize states for accumulation
        self.add_state("vertices_pred", default=[], dist_reduce_fx=None)
        self.add_state("vertices_gt", default=[], dist_reduce_fx=None)

    def update(self, vertices_pred: Tensor, vertices_gt: Tensor) -> None:
        """Update metric states with predictions and targets.

        Args:
            vertices_pred: Predicted vertices tensor of shape (T, V, 3) where T is number of frames,
                V is number of vertices, and 3 represents XYZ coordinates
            vertices_gt: Ground truth vertices tensor of shape (T, V, 3)
        """
        self.vertices_pred.append(vertices_pred)
        self.vertices_gt.append(vertices_gt)

    def compute(self) -> Tensor:
        """Compute the Lip Vertex Error over all accumulated states.

        Returns:
            Tensor: A scalar tensor with the mean LVE value
        """
        vertices_pred = torch.cat(self.vertices_pred, dim=0)
        vertices_gt = torch.cat(self.vertices_gt, dim=0)
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

        """
        return self._plot(val, ax)