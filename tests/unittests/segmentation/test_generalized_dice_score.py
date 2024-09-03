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

import pytest
import torch
from monai.metrics.generalized_dice import compute_generalized_dice
from torchmetrics.functional.segmentation.generalized_dice import generalized_dice_score
from torchmetrics.segmentation.generalized_dice import GeneralizedDiceScore

from unittests import BATCH_SIZE, NUM_BATCHES, NUM_CLASSES, _Input
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

seed_all(42)

_inputs1 = _Input(
    preds=torch.randint(0, 2, (NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, 16)),
    target=torch.randint(0, 2, (NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, 16)),
)
_inputs2 = _Input(
    preds=torch.randint(0, 2, (NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, 32, 32)),
    target=torch.randint(0, 2, (NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, 32, 32)),
)
_inputs3 = _Input(
    preds=torch.randint(0, NUM_CLASSES, (NUM_BATCHES, BATCH_SIZE, 32, 32)),
    target=torch.randint(0, NUM_CLASSES, (NUM_BATCHES, BATCH_SIZE, 32, 32)),
)


def _reference_generalized_dice(
    preds: torch.Tensor,
    target: torch.Tensor,
    input_format: str,
    include_background: bool = True,
    reduce: bool = True,
):
    """Calculate reference metric for `MeanIoU`."""
    if input_format == "index":
        preds = torch.nn.functional.one_hot(preds, num_classes=NUM_CLASSES).movedim(-1, 1)
        target = torch.nn.functional.one_hot(target, num_classes=NUM_CLASSES).movedim(-1, 1)
    val = compute_generalized_dice(preds, target, include_background=include_background)
    if reduce:
        val = val.mean()
    return val


@pytest.mark.parametrize(
    "preds, target, input_format",
    [
        (_inputs1.preds, _inputs1.target, "one-hot"),
        (_inputs2.preds, _inputs2.target, "one-hot"),
        (_inputs3.preds, _inputs3.target, "index"),
    ],
)
@pytest.mark.parametrize("include_background", [True, False])
class TestGeneralizedDiceScore(MetricTester):
    """Test class for `MeanIoU` metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_generalized_dice_class(self, preds, target, input_format, include_background, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=GeneralizedDiceScore,
            reference_metric=partial(
                _reference_generalized_dice,
                input_format=input_format,
                include_background=include_background,
                reduce=True,
            ),
            metric_args={
                "num_classes": NUM_CLASSES,
                "include_background": include_background,
                "input_format": input_format,
            },
        )

    def test_generalized_dice_functional(self, preds, target, input_format, include_background):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=generalized_dice_score,
            reference_metric=partial(
                _reference_generalized_dice,
                input_format=input_format,
                include_background=include_background,
                reduce=False,
            ),
            metric_args={
                "num_classes": NUM_CLASSES,
                "include_background": include_background,
                "per_class": False,
                "input_format": input_format,
            },
        )
