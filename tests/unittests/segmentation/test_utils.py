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
import pytest
import torch
from monai.metrics.utils import get_code_to_measure_table
from monai.metrics.utils import get_edge_surface_distance as monai_get_edge_surface_distance
from monai.metrics.utils import get_mask_edges as monai_get_mask_edges
from monai.metrics.utils import get_surface_distance as monai_get_surface_distance
from scipy.ndimage import binary_erosion as scibinary_erosion
from scipy.ndimage import distance_transform_cdt as scidistance_transform_cdt
from scipy.ndimage import distance_transform_edt as scidistance_transform_edt
from scipy.ndimage import generate_binary_structure as scigenerate_binary_structure

from torchmetrics.functional.segmentation.utils import (
    binary_erosion,
    distance_transform,
    edge_surface_distance,
    generate_binary_structure,
    get_neighbour_tables,
    mask_edges,
    surface_distance,
)


@pytest.mark.parametrize("rank", [2, 3, 4])
@pytest.mark.parametrize("connectivity", [1, 2, 3])
def test_generate_binary_structure(rank, connectivity):
    """Test the generate binary structure function."""
    structure = generate_binary_structure(rank, connectivity)
    scistucture = scigenerate_binary_structure(rank, connectivity)
    assert torch.allclose(structure, torch.from_numpy(scistucture))


@pytest.mark.parametrize(
    "case",
    [
        torch.ones(3, 1),
        torch.ones(5, 1),
        torch.ones(3, 3),
        torch.tensor([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]),
        torch.tensor([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]),
        torch.tensor([
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ]),
        torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 0], [0, 0, 1, 0, 1], [0, 1, 1, 1, 0], [0, 1, 1, 1, 1]]),
        torch.randint(2, (5, 5)),
        torch.randint(2, (20, 20)),
        torch.ones(5, 5, 5),
        torch.randint(2, (5, 5, 5)),
        torch.randint(2, (20, 20, 20)),
    ],
)
@pytest.mark.parametrize("border_value", [0, 1])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_binary_erosion(case, border_value, device):
    """Test the binary erosion function.

    Cases taken from:
    https://github.com/scipy/scipy/blob/v1.11.1/scipy/ndimage/tests/test_morphology.py

    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device not available.")
    scierosion = scibinary_erosion(case, border_value=border_value)
    erosion = binary_erosion(case.unsqueeze(0).unsqueeze(0).to(device), border_value=border_value)
    assert torch.allclose(erosion.cpu(), torch.from_numpy(scierosion).byte())


@pytest.mark.parametrize(
    ("arguments", "error", "match"),
    [
        (([0, 1, 2, 3],), TypeError, "Expected argument `image` to be of type Tensor.*"),
        ((torch.ones(3, 3),), ValueError, "Expected argument `image` to be of rank 4 or 5 but.*"),
        ((torch.randint(3, (1, 1, 5, 5)),), ValueError, "Input x should be binarized"),
    ],
)
def test_binary_erosion_error(arguments, error, match):
    """Test that binary erosion raises an error when the input is not binary."""
    with pytest.raises(error, match=match):
        binary_erosion(*arguments)


@pytest.mark.parametrize(
    "case",
    [
        torch.tensor([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]),
        torch.tensor([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]),
        torch.tensor([
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ]),
    ],
)
@pytest.mark.parametrize("metric", ["euclidean", "chessboard", "taxicab"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_distance_transform(case, metric, device):
    """Test the distance transform function.

    Cases taken from:
    https://github.com/scipy/scipy/blob/v1.11.1/scipy/ndimage/tests/test_morphology.py

    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device not available.")
    distance = distance_transform(case.to(device), metric=metric)
    if metric == "euclidean":
        scidistance = scidistance_transform_edt(case)
    else:
        scidistance = scidistance_transform_cdt(case, metric=metric)
    assert torch.allclose(distance.cpu(), torch.from_numpy(scidistance).to(distance.dtype))


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("spacing", [1, 2])
def test_neighbour_table(dim, spacing):
    """Test the table for surface score function."""
    spacing = dim * (spacing,)
    ref_table, ref_kernel = get_code_to_measure_table(spacing)
    table, kernel = get_neighbour_tables(spacing)

    assert torch.allclose(ref_table.float(), table)
    assert torch.allclose(ref_kernel, kernel)


@pytest.mark.parametrize(
    "cases",
    [
        (
            torch.tensor(
                [[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]], dtype=torch.bool
            ),
            torch.tensor(
                [[1, 1, 1, 1, 0], [1, 0, 0, 1, 0], [1, 0, 0, 1, 0], [1, 0, 0, 1, 0], [1, 1, 1, 1, 0]], dtype=torch.bool
            ),
        ),
        (torch.randint(0, 2, (5, 5), dtype=torch.bool), torch.randint(0, 2, (5, 5), dtype=torch.bool)),
        (torch.randint(0, 2, (50, 50), dtype=torch.bool), torch.randint(0, 2, (50, 50), dtype=torch.bool)),
    ],
)
@pytest.mark.parametrize("distance_metric", ["euclidean", "chessboard", "taxicab"])
@pytest.mark.parametrize("spacing", [1, 2])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_surface_distance(cases, distance_metric, spacing, device):
    """Test the surface distance function."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device not available.")
    if spacing != 1 and distance_metric != "euclidean":
        pytest.skip("Only euclidean distance is supported for spacing != 1 in reference")
    preds, target = cases
    spacing = 2 * [spacing]
    res = surface_distance(preds.to(device), target.to(device), distance_metric=distance_metric, spacing=spacing)
    reference_res = monai_get_surface_distance(
        preds.numpy(), target.numpy(), distance_metric=distance_metric, spacing=spacing
    )
    assert torch.allclose(res.cpu(), torch.from_numpy(reference_res).to(res.dtype))


@pytest.mark.parametrize(
    "cases",
    [
        (torch.randint(0, 2, (5, 5), dtype=torch.bool), torch.randint(0, 2, (5, 5), dtype=torch.bool)),
        (torch.randint(0, 2, (50, 50), dtype=torch.bool), torch.randint(0, 2, (50, 50), dtype=torch.bool)),
        (torch.randint(0, 2, (50, 50, 50), dtype=torch.bool), torch.randint(0, 2, (50, 50, 50), dtype=torch.bool)),
    ],
)
@pytest.mark.parametrize("spacing", [None, 1, 2])
@pytest.mark.parametrize("crop", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mask_edges(cases, spacing, crop, device):
    """Test the mask edges function."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device not available.")
    preds, target = cases
    if spacing is not None:
        spacing = preds.ndim * (spacing,)
    res = mask_edges(preds.to(device), target.to(device), spacing=spacing, crop=crop)
    reference_res = monai_get_mask_edges(preds, target, spacing=spacing, crop=crop)

    for r1, r2 in zip(res, reference_res):
        assert torch.allclose(r1.cpu().float(), torch.from_numpy(r2).float())


@pytest.mark.parametrize(
    "cases",
    [
        (
            torch.tensor(
                [[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]], dtype=torch.bool
            ),
            torch.tensor(
                [[1, 1, 1, 1, 0], [1, 0, 0, 1, 0], [1, 0, 0, 1, 0], [1, 0, 0, 1, 0], [1, 1, 1, 1, 0]], dtype=torch.bool
            ),
        ),
        (torch.randint(0, 2, (5, 5), dtype=torch.bool), torch.randint(0, 2, (5, 5), dtype=torch.bool)),
        (torch.randint(0, 2, (50, 50), dtype=torch.bool), torch.randint(0, 2, (50, 50), dtype=torch.bool)),
    ],
)
@pytest.mark.parametrize("distance_metric", ["euclidean", "chessboard", "taxicab"])
@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("spacing", [None, 1, 2])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_edge_surface_distance(cases, distance_metric, symmetric, spacing, device):
    """Test the edge surface distance function."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device not available.")
    if spacing == 2 and distance_metric != "euclidean":
        pytest.skip("Only euclidean distance is supported for spacing != 1 in reference")
    preds, target = cases
    if spacing is not None:
        spacing = preds.ndim * [spacing]

    res = edge_surface_distance(
        preds.to(device), target.to(device), spacing=spacing, distance_metric=distance_metric, symmetric=symmetric
    )
    _, reference_res, _ = monai_get_edge_surface_distance(
        preds,
        target,
        spacing=tuple(spacing) if spacing is not None else spacing,
        distance_metric=distance_metric,
        symmetric=symmetric,
    )

    if symmetric:
        assert torch.allclose(res[0].cpu(), reference_res[0].to(res[0].dtype))
        assert torch.allclose(res[1].cpu(), reference_res[1].to(res[1].dtype))
    else:
        assert torch.allclose(res.cpu(), reference_res[0].to(res.dtype))
