# Copyright The PyTorch Lightning team.
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

from torchmetrics.detection.ciou import CompleteIntersectionOverUnion
from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_13
from unittests.helpers.testers import MetricTester

preds = torch.Tensor(
    [
        [296.55, 93.96, 314.97, 152.79],
        [328.94, 97.05, 342.49, 122.98],
        [356.62, 95.47, 372.33, 147.55],
        [464.08, 105.09, 495.74, 146.99],
        [276.11, 103.84, 291.44, 150.72],
    ]
)
target = torch.Tensor(
    [
        [61.87, 276.25, 358.29, 379.43],
        [2.75, 3.66, 162.15, 316.06],
        [295.55, 93.96, 313.97, 152.79],
        [326.94, 97.05, 340.49, 122.98],
        [356.62, 95.47, 372.33, 147.55],
        [462.08, 105.09, 493.74, 146.99],
        [277.11, 103.84, 292.44, 150.72],
    ]
)

ciou = torch.Tensor(
    [
        [0.0000, 0.0000, 0.8970, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.7428, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.8812, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.8775],
    ]
)


_pytest_condition = not (_TORCHVISION_AVAILABLE and _TORCHVISION_GREATER_EQUAL_0_13)


@pytest.mark.skipif(_pytest_condition, reason="test requires that torchvision=>0.13.0 is installed")
class TestCompleteIntersectionOverUnion(MetricTester):
    """Test the Complete Intersection over Union metric for object detection predictions."""

    atol = 1e-1

    @pytest.mark.parametrize("ddp", [False, True])
    def test_ciou(self, ddp):
        """Test modular implementation for correctness."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,  # Note: we fail this test because len(preds) != len(target)
            target=target,
            metric_class=CompleteIntersectionOverUnion,
            sk_metric=ciou,
            dist_sync_on_step=False,
            check_batch=False,
            metric_args={},
        )
