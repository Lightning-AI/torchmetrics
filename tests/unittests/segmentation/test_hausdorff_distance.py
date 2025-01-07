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
from functools import partial
from typing import Any

import pytest
import torch
from monai.metrics.hausdorff_distance import compute_hausdorff_distance as monai_hausdorff_distance

from torchmetrics.functional.segmentation.hausdorff_distance import hausdorff_distance
from torchmetrics.segmentation.hausdorff_distance import HausdorffDistance
from unittests import NUM_BATCHES, _Input
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

seed_all(42)
BATCH_SIZE = 4  # use smaller than normal batch size to reduce test time
NUM_CLASSES = 3  # use smaller than normal class size to reduce test time

_inputs1 = _Input(
    preds=torch.randint(0, 2, (NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, 16, 16)),
    target=torch.randint(0, 2, (NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, 16, 16)),
)
_inputs2 = _Input(
    preds=torch.randint(0, NUM_CLASSES, (NUM_BATCHES, BATCH_SIZE, 32, 32)),
    target=torch.randint(0, NUM_CLASSES, (NUM_BATCHES, BATCH_SIZE, 32, 32)),
)


def reference_metric(preds, target, input_format, reduce, **kwargs: Any):
    """Reference implementation of metric."""
    if input_format == "index":
        preds = torch.nn.functional.one_hot(preds, num_classes=NUM_CLASSES).movedim(-1, 1)
        target = torch.nn.functional.one_hot(target, num_classes=NUM_CLASSES).movedim(-1, 1)
    score = monai_hausdorff_distance(preds, target, **kwargs)
    return score.mean() if reduce else score


@pytest.mark.parametrize("inputs, input_format", [(_inputs1, "one-hot"), (_inputs2, "index")])
@pytest.mark.parametrize("distance_metric", ["euclidean", "chessboard", "taxicab"])
@pytest.mark.parametrize("directed", [True, False])
@pytest.mark.parametrize("spacing", [None, [2, 2]])
class TestHausdorffDistance(MetricTester):
    """Test class for `HausdorffDistance` metric."""

    atol = 1e-5

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_hausdorff_distance_class(self, inputs, input_format, distance_metric, directed, spacing, ddp):
        """Test class implementation of metric."""
        if spacing is not None and distance_metric != "euclidean":
            pytest.skip("Spacing is only supported for Euclidean distance metric.")
        preds, target = inputs
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=HausdorffDistance,
            reference_metric=partial(
                reference_metric,
                input_format=input_format,
                distance_metric=distance_metric,
                directed=directed,
                spacing=spacing,
                reduce=True,
            ),
            metric_args={
                "num_classes": NUM_CLASSES,
                "distance_metric": distance_metric,
                "directed": directed,
                "spacing": spacing,
                "input_format": input_format,
            },
        )

    def test_hausdorff_distance_functional(self, inputs, input_format, distance_metric, directed, spacing):
        """Test functional implementation of metric."""
        if spacing is not None and distance_metric != "euclidean":
            pytest.skip("Spacing is only supported for Euclidean distance metric.")
        preds, target = inputs
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=hausdorff_distance,
            reference_metric=partial(
                reference_metric,
                input_format=input_format,
                distance_metric=distance_metric,
                directed=directed,
                spacing=spacing,
                reduce=False,
            ),
            metric_args={
                "num_classes": NUM_CLASSES,
                "distance_metric": distance_metric,
                "directed": directed,
                "spacing": spacing,
                "input_format": input_format,
            },
        )


def test_hausdorff_distance_raises_error():
    """Check that metric raises appropriate errors."""
    preds, target = _inputs1
