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
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn.functional import conv2d, conv3d, pad, unfold
from typing_extensions import Literal

from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.imports import _SCIPY_AVAILABLE


def check_if_binarized(x: Tensor) -> bool:
    """Check if the input is binarized."""
    if not torch.all(x.bool() == x):
        raise ValueError("Input x should be binarized")


def generate_binary_structure(rank: int, connectivity: int) -> Tensor:
    """Translated version of the function from scipy.ndimage.morphology.

    Args:
        rank: The rank of the structuring element.
        connectivity: The number of neighbors connected to a given pixel.

    Returns:
        The structuring element.

    Examples::
        >>> from torchmetrics.functional.segmentation.helper import generate_binary_structure
        >>> import torch
        >>> generate_binary_structure(2, 1)
        tensor([[False,  True, False],
                [ True,  True,  True],
                [False,  True, False]])
        >>> generate_binary_structure(2, 2)
        tensor([[True,  True,  True],
                [True,  True,  True],
                [True,  True,  True]])
        >>> generate_binary_structure(2, 3)
        tensor([[[False,  True, False],
                 [True,  True,  True],
                 [False,  True, False]],

                [[True,  True,  True],
                 [True,  True,  True],
                 [True,  True,  True]],

                [[False,  True, False],
                 [True,  True,  True],
                 [False,  True, False]]])

    """
    if connectivity < 1:
        connectivity = 1
    if rank < 1:
        return torch.tensor([1], dtype=torch.uint8)
    grids = torch.meshgrid([torch.arange(3) for _ in range(rank)], indexing="ij")
    output = torch.abs(torch.stack(grids, dim=0) - 1)
    output = torch.sum(output, dim=0)
    return output <= connectivity


def binary_erosion(
    image: Tensor, structure: Optional[Tensor] = None, origin: Tuple[int, int] = (1, 1), border_value: int = 0
) -> Tensor:
    """Binary erosion of a tensor image.

    Implementation inspired by answer to this question: https://stackoverflow.com/questions/56235733/

    Args:
        image: The image to be eroded, must be a binary tensor with shape ``(batch_size, channels, height, width)``.
        structure: The structuring element used for the erosion. If no structuring element is provided, an element
            is generated with a square connectivity equal to one.
        origin: The origin of the structuring element.
        border_value: The value to be used for the border.

    Examples::
        >>> from torchmetrics.functional.segmentation.helper import binary_erosion
        >>> import torch
        >>> image = torch.tensor([[[[0, 0, 0, 0, 0],
        ...                         [0, 1, 1, 1, 0],
        ...                         [0, 1, 1, 1, 0],
        ...                         [0, 1, 1, 1, 0],
        ...                         [0, 0, 0, 0, 0]]]])
        >>> binary_erosion(image)
        tensor([[[[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]]], dtype=torch.uint8)
        >>> binary_erosion(image, structure=torch.ones(4, 4))
        tensor([[[[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]]], dtype=torch.uint8)

    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Expected argument `image` to be of type Tensor but found {type(image)}")
    if image.ndim != 4:
        raise ValueError(f"Expected argument `image` to be of rank 4 but found rank {image.ndim}")
    check_if_binarized(image)

    # construct the structuring element if not provided
    if structure is None:
        structure = generate_binary_structure(image.ndim - 2, 1).int()
    check_if_binarized(structure)

    # first pad the image to have correct unfolding; here is where the origins is used
    image_pad = pad(
        image,
        [origin[0], structure.shape[0] - origin[0] - 1, origin[1], structure.shape[1] - origin[1] - 1],
        mode="constant",
        value=border_value,
    )
    # Unfold the image to be able to perform operation on neighborhoods
    image_unfold = unfold(image_pad.float(), kernel_size=structure.shape)

    strel_flatten = torch.flatten(structure).unsqueeze(0).unsqueeze(-1)
    sums = image_unfold - strel_flatten.int()

    # Take minimum over the neighborhood
    result, _ = sums.min(dim=1)

    # Reshape the image to recover initial shape
    return (torch.reshape(result, image.shape) + 1).byte()


def distance_transform(
    x: Tensor,
    sampling: Optional[List[float]] = None,
    metric: Literal["euclidean", "chessboard", "taxicab"] = "euclidean",
    engine: Literal["pytorch", "scipy"] = "pytorch",
) -> Tensor:
    """Calculate distance transform of a binary tensor.

    This function calculates the distance transfrom of a binary tensor, replacing each foreground pixel with the
    distance to the closest background pixel. The distance is calculated using the euclidean, chessboard or taxicab
    distance.

    Args:
        x: The binary tensor to calculate the distance transform of.
        sampling: Only relevant when distance is calculated using the euclidean distance. The sampling referes to the
            pixel spacing in the image, i.e. the distance between two adjacent pixels. If not provided, the pixel
            spacing is assumed to be 1.
        metric: The distance to use for the distance transform. Can be one of ``"euclidean"``, ``"chessboard"``
            or ``"taxicab"``.
        engine: The engine to use for the distance transform. Can be one of ``["pytorch", "scipy"]``. In general,
            the ``pytorch`` engine is faster, but the ``scipy`` engine is more memory efficient.

    Returns:
        The distance transform of the input tensor.

    Examples::
        >>> from torchmetrics.functional.segmentation.helper import distance_transform
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

    if engine == "pytorch":
        x = x.float()
        # calculate distance from every pixel to every other pixel
        n_zero = torch.sum(x == 0)
        y = x.view(-1, 1, 1) - x
        y[y == -1] = 0
        i, j, k = torch.where(y)

        # calculate distance
        h, w = x.shape
        if metric == "euclidean":
            dis = ((sampling[0] * abs(i // h - j)) ** 2 + (sampling[1] * abs(i % w - k)) ** 2).sqrt()
        if metric == "chessboard":
            dis = torch.max(sampling[0] * abs(i // h - j), sampling[1] * abs(i % w - k)).float()
        if metric == "taxicab":
            dis = sampling[0] * abs(i // h - j) + sampling[1] * abs(i % w - k).float()

        # select only the closest distance
        dis = dis.reshape(-1, n_zero)
        dis, _ = torch.min(dis, dim=1)
        i = i[::n_zero]

        z = torch.zeros_like(x).view(-1)
        z[i] = dis
        return z.view(x.shape)

    if not _SCIPY_AVAILABLE:
        raise ValueError(
            "The `scipy` engine requires `scipy` to be installed. Either install `scipy` or use the `pytorch` engine."
        )
    from scipy import ndimage

    if metric == "euclidean":
        return ndimage.distance_transform_edt(x.cpu().numpy(), sampling)
    return ndimage.distance_transform_cdt(x.cpu().numpy(), sampling, metric=metric)


def create_table_neighbour_code_to_contour_length(spacing: List[int], device: Optional[torch.device] = None) -> Tensor:
    """Create a table that maps neighbour codes to the contour length of the corresponding contour."""
    ...


def create_table_neighbour_code_to_surface_area(spacing: List[int], device: Optional[torch.device] = None) -> Tensor:
    """Create a table that maps neighbour codes to the surface area of the corresponding surface."""
    ...


def get_code_to_measure_table(spacing: List[int], device: Optional[torch.device] = None) -> Tensor:
    """Create a table that maps neighbour codes to the measure of the contour length or surface area."""
    spatial_dims = len(spacing)
    if spatial_dims == 2:
        return create_table_neighbour_code_to_contour_length(spacing, device)
    return create_table_neighbour_code_to_surface_area(spacing, device)


def get_mask_edges(
    preds: Tensor,
    target: Tensor,
    label_idx: int = 1,
    crop: bool = True,
    spacing: List[int] | None = None,
) -> Tuple[Tensor, Tensor]:
    """Get the edges of the masks in the input tensors."""
    _check_same_shape(preds, target)

    if crop:
        or_val = preds | target
        if not or_val.any():
            preds, target = torch.zeros_like(preds), torch.zeros_like(target)
            return (preds, target) if spacing is None else (preds, target, preds, target)
        [preds.unsqueeze(0), target.unsqueeze(0), or_val.unsqueeze(0)]
        if spacing is None:
            ...

    if spacing is None:
        edges_preds = binary_erosion(preds) ^ preds
        edges_target = binary_erosion(target) ^ target
        return edges_preds, edges_target

    code_to_area_table, k = get_code_to_measure_table(spacing, device=preds.device)
    spatial_dims = len(spacing)
    conv_operator = conv2d if spatial_dims == 2 else conv3d
    volume = torch.stack([preds.unsqueeze(0), target.unsqueeze(0)], dim=0).float()
    code_preds, code_target = conv_operator(volume, k.to(volume))

    # edges
    all_ones = len(code_to_area_table) - 1
    edges_preds = (code_preds != 0) & (code_preds != all_ones)
    edges_target = (code_target != 0) & (code_target != all_ones)

    # areas of edges
    areas_preds = torch.index_select(code_to_area_table, 0, code_preds.view(-1).int()).view_as(code_preds)
    areas_target = torch.index_select(code_to_area_table, 0, code_target.view(-1).int()).view_as(code_target)
    return edges_preds[0], edges_target[0], areas_preds[0], areas_target[0]


def get_surface_distance(
    preds: Tensor,
    target: Tensor,
    distance_metric: Literal["euclidean", "chessboard", "taxicab"] = "euclidean",
    spacing: Optional[Tensor] = None,
) -> Tensor:
    """Calculate the surface distance between two binary volumes."""
    if not torch.any(target):
        dis = torch.inf * torch.ones_like(target)
    else:
        if not torch.any(preds):
            dis = torch.inf * torch.ones_like(preds)
            return dis[target]
        if distance_metric == "euclidean":
            dis = torch.cdist(preds, target, p=2)
        elif distance_metric == "chessboard":
            dis = torch.cdist(preds, target, p=float("inf"))
    return dis[preds]
