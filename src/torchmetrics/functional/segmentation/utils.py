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
from typing import List, Optional, Union

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.utilities.imports import _SCIPY_AVAILABLE


def _check_if_binarized(x: Tensor) -> None:
    """Check if the input is binarized.

    Example:
        >>> from torchmetrics.functional.segmentation.utils import check_if_binarized
        >>> import torch
        >>> check_if_binarized(torch.tensor([0, 1, 1, 0]))

    """
    if not torch.all(x.bool() == x):
        raise ValueError("Input x should be binarized")


def distance_transform(
    x: Tensor,
    sampling: Optional[Union[Tensor, List[float]]] = None,
    metric: Literal["euclidean", "chessboard", "taxicab"] = "euclidean",
    engine: Literal["pytorch", "scipy"] = "pytorch",
) -> Tensor:
    """Calculate distance transform of a binary tensor.

    This function calculates the distance transform of a binary tensor, replacing each foreground pixel with the
    distance to the closest background pixel. The distance is calculated using the euclidean, chessboard or taxicab
    distance.

    Args:
        x: The binary tensor to calculate the distance transform of.
        sampling: Only relevant when distance is calculated using the euclidean distance. The sampling refers to the
            pixel spacing in the image, i.e. the distance between two adjacent pixels. If not provided, the pixel
            spacing is assumed to be 1.
        metric: The distance to use for the distance transform. Can be one of ``"euclidean"``, ``"chessboard"``
            or ``"taxicab"``.
        engine: The engine to use for the distance transform. Can be one of ``["pytorch", "scipy"]``. In general,
            the ``pytorch`` engine is faster, but the ``scipy`` engine is more memory efficient.

    Returns:
        The distance transform of the input tensor.

    Examples::
        >>> from torchmetrics.functional.segmentation.utils import distance_transform
        >>> import torch
        >>> x = torch.tensor([[0, 0, 0, 0, 0],
        ...                   [0, 1, 1, 1, 0],
        ...                   [0, 1, 1, 1, 0],
        ...                   [0, 1, 1, 1, 0],
        ...                   [0, 0, 0, 0, 0]])
        >>> distance_transform(x)
        tensor([[0., 0., 0., 0., 0.],
                [0., 1., 1., 1., 0.],
                [0., 1., 2., 1., 0.],
                [0., 1., 1., 1., 0.],
                [0., 0., 0., 0., 0.]])

    """
    if not isinstance(x, Tensor):
        raise ValueError(f"Expected argument `x` to be of type `torch.Tensor` but got `{type(x)}`.")
    if x.ndim != 2:
        raise ValueError(f"Expected argument `x` to be of rank 2 but got rank `{x.ndim}`.")
    if sampling is not None and not isinstance(sampling, list):
        raise ValueError(
            f"Expected argument `sampling` to either be `None` or of type `list` but got `{type(sampling)}`."
        )
    if metric not in ["euclidean", "chessboard", "taxicab"]:
        raise ValueError(
            f"Expected argument `metric` to be one of `['euclidean', 'chessboard', 'taxicab']` but got `{metric}`."
        )
    if engine not in ["pytorch", "scipy"]:
        raise ValueError(f"Expected argument `engine` to be one of `['pytorch', 'scipy']` but got `{engine}`.")

    if sampling is None:
        sampling = [1, 1]
    else:
        if len(sampling) != 2:
            raise ValueError(f"Expected argument `sampling` to have length 2 but got length `{len(sampling)}`.")

    if engine == "pytorch":
        x = x.float()
        # calculate distance from every foreground pixel to every background pixel
        i0, j0 = torch.where(x == 0)
        i1, j1 = torch.where(x == 1)
        dis_row = (i1.view(-1, 1) - i0.view(1, -1)).abs()
        dis_col = (j1.view(-1, 1) - j0.view(1, -1)).abs()

        # # calculate distance
        h, _ = x.shape
        if metric == "euclidean":
            dis = ((sampling[0] * dis_row) ** 2 + (sampling[1] * dis_col) ** 2).sqrt()
        if metric == "chessboard":
            dis = torch.max(sampling[0] * dis_row, sampling[1] * dis_col).float()
        if metric == "taxicab":
            dis = (sampling[0] * dis_row + sampling[1] * dis_col).float()

        # select only the closest distance
        mindis, _ = torch.min(dis, dim=1)
        z = torch.zeros_like(x).view(-1)
        z[i1 * h + j1] = mindis
        return z.view(x.shape)

    if not _SCIPY_AVAILABLE:
        raise ValueError(
            "The `scipy` engine requires `scipy` to be installed. Either install `scipy` or use the `pytorch` engine."
        )
    from scipy import ndimage

    if metric == "euclidean":
        return ndimage.distance_transform_edt(x.cpu().numpy(), sampling)
    return ndimage.distance_transform_cdt(x.cpu().numpy(), sampling, metric=metric)


def surface_distance(
    preds: Tensor,
    target: Tensor,
    distance_metric: Literal["euclidean", "chessboard", "taxicab"] = "euclidean",
    spacing: Optional[Union[Tensor, List[float]]] = None,
) -> Tensor:
    """Calculate the surface distance between two binary edge masks.

    May return infinity if the predicted mask is empty and the target mask is not, or vice versa.

    Args:
        preds: The predicted binary edge mask.
        target: The target binary edge mask.
        distance_metric: The distance metric to use. One of `["euclidean", "chessboard", "taxicab"]`.
        spacing: The spacing between pixels along each spatial dimension.

    Returns:
        A tensor with length equal to the number of edges in predictions e.g. `preds.sum()`. Each element is the
        distance from the corresponding edge in `preds` to the closest edge in `target`.

    Example::
        >>> import torch
        >>> from torchmetrics.functional.segmentation.utils import surface_distance
        >>> preds = torch.tensor([[1, 1, 1, 1, 1],
        ...                       [1, 0, 0, 0, 1],
        ...                       [1, 0, 0, 0, 1],
        ...                       [1, 0, 0, 0, 1],
        ...                       [1, 1, 1, 1, 1]], dtype=torch.bool)
        >>> target = torch.tensor([[1, 1, 1, 1, 0],
        ...                        [1, 0, 0, 1, 0],
        ...                        [1, 0, 0, 1, 0],
        ...                        [1, 0, 0, 1, 0],
        ...                        [1, 1, 1, 1, 0]], dtype=torch.bool)
        >>> surface_distance(preds, target, distance_metric="euclidean", spacing=[1, 1])
        tensor([0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0., 1.])

    """
    if not (preds.dtype == torch.bool and target.dtype == torch.bool):
        raise ValueError(f"Expected both inputs to be of type `torch.bool`, but got {preds.dtype} and {target.dtype}.")

    if not torch.any(target):
        dis = torch.inf * torch.ones_like(target)
    else:
        if not torch.any(preds):
            dis = torch.inf * torch.ones_like(preds)
            return dis[target]
        dis = distance_transform(~target, sampling=spacing, metric=distance_metric)
    return dis[preds]
