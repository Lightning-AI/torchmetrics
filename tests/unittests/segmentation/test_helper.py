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
from scipy.ndimage.morphology import binary_erosion as scibinary_erosion
from scipy.ndimage.morphology import distance_transform_cdt as scidistance_transform_cdt
from scipy.ndimage.morphology import distance_transform_edt as scidistance_transform_edt
from scipy.ndimage.morphology import generate_binary_structure as scigenerate_binary_structure

from torchmetrics.functional.segmentation.helper import binary_erosion, distance_transform, generate_binary_structure


@pytest.mark.parametrize("rank", [2, 3, 4])
@pytest.mark.parametrize("connectivity", [1, 2, 3])
def test_generate_binary_structure(rank, connectivity):
    """Test the generate binary structure function."""
    stucture = generate_binary_structure(rank, connectivity)
    scistucture = scigenerate_binary_structure(rank, connectivity)
    assert torch.allclose(stucture, torch.from_numpy(scistucture))


@pytest.mark.parametrize(
    "case",
    [
        torch.ones(3, 1),
        torch.ones(5, 1),
        torch.ones(3, 3),
        torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ),
        torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]
        ),
        torch.tensor(
            [
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        ),
    ],
)
@pytest.mark.parametrize("border_value", [0, 1])
def test_binary_erosion(case, border_value):
    """Test the binary erosion function.

    Cases taken from:
    https://github.com/scipy/scipy/blob/v1.11.1/scipy/ndimage/tests/test_morphology.py

    """
    scierosion = scibinary_erosion(case, border_value=border_value)
    erosion = binary_erosion(case.unsqueeze(0).unsqueeze(0), border_value=border_value)
    assert torch.allclose(erosion, torch.from_numpy(scierosion).float())


@pytest.mark.parametrize(
    ("arguments", "error", "match"),
    [
        (([0, 1, 2, 3],), TypeError, "Expected argument `image` to be of type Tensor.*"),
        ((torch.ones(3, 3),), ValueError, "Expected argument `image` to be of rank 4 but.*"),
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
        torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ),
        torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]
        ),
        torch.tensor(
            [
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        ),
    ],
)
@pytest.mark.parametrize("metric", ["euclidean", "chessboard", "taxicab"])
def test_distance_transform(case, metric):
    """Test the distance transform function.

    Cases taken from:
    https://github.com/scipy/scipy/blob/v1.11.1/scipy/ndimage/tests/test_morphology.py

    """
    distance = distance_transform(case, metric=metric)
    if metric == "euclidean":
        scidistance = scidistance_transform_edt(case)
    else:
        scidistance = scidistance_transform_cdt(case, metric=metric)
    assert torch.allclose(distance, torch.from_numpy(scidistance).float())
