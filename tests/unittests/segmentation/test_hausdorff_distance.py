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
from skimage.metrics import hausdorff_distance as skimage_hausdorff_distance
from torchmetrics.functional.segmentation.hausdorff_distance import hausdorff_distance
from torchmetrics.segmentation.hausdorff_distance import HausdorffDistance

from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester
from unittests.segmentation.inputs import _Input

seed_all(42)


preds = torch.tensor(
    [
        [[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]],
        [[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]],
        [[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]],
        [[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]],
    ],
    dtype=torch.bool,
)

target = torch.tensor(
    [
        [[1, 1, 1, 1, 0], [1, 0, 0, 1, 0], [1, 0, 0, 1, 0], [1, 0, 0, 1, 0], [1, 1, 1, 1, 0]],
        [[1, 1, 1, 1, 0], [1, 0, 0, 1, 0], [1, 0, 0, 1, 0], [1, 0, 0, 1, 0], [1, 1, 1, 1, 0]],
        [[1, 1, 1, 1, 0], [1, 0, 0, 1, 0], [1, 0, 0, 1, 0], [1, 0, 0, 1, 0], [1, 1, 1, 1, 0]],
        [[1, 1, 1, 1, 0], [1, 0, 0, 1, 0], [1, 0, 0, 1, 0], [1, 0, 0, 1, 0], [1, 1, 1, 1, 0]],
    ],
    dtype=torch.bool,
)

_inputs = _Input(preds=preds, target=target)


# Wrapper that converts to numpy to avoid Torch-to-numpy functional issues
def torch_skimage_hausdorff_distance(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    out = skimage_hausdorff_distance(p.numpy(), t.numpy())
    return torch.tensor([out])


@pytest.mark.parametrize(
    "preds, target",
    [
        (_inputs.preds, _inputs.target),
    ],
)
@pytest.mark.parametrize(
    "distance_metric",
    ["euclidean", "chessboard", "taxicab"],
)
class TestHausdorffDistance(MetricTester):
    """Test class for `HausdorffDistance` metric."""

    atol = 1e-5

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_hausdorff_distance_class(self, preds, target, distance_metric, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=HausdorffDistance,
            reference_metric=torch_skimage_hausdorff_distance,
            metric_args={"distance_metric": distance_metric, "spacing": None},
        )

    def test_hausdorff_distance_functional(self, preds, target, distance_metric):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=hausdorff_distance,
            reference_metric=torch_skimage_hausdorff_distance,
            metric_args={"distance_metric": distance_metric, "spacing": None},
        )


def test_hausdorff_distance_functional_raises_invalid_task():
    """Check that metric rejects continuous-valued inputs."""
    preds, target = _inputs
    with pytest.raises(ValueError, match=r"Expected *"):
        hausdorff_distance(preds, target)


@pytest.mark.parametrize(
    "distance_metric",
    ["euclidean", "chessboard", "taxicab"],
)
def test_hausdorff_distance_is_symmetric(distance_metric):
    """Check that the metric functional is symmetric."""
    for p, t in zip(_inputs.preds, _inputs.target):
        assert torch.allclose(
            hausdorff_distance(p, t, distance_metric),
            hausdorff_distance(t, p, distance_metric),
        )
