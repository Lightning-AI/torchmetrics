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
from lightning_utilities.core.imports import RequirementCache
from monai.metrics.generalized_dice import compute_generalized_dice

from torchmetrics.functional.segmentation.generalized_dice import generalized_dice_score
from torchmetrics.segmentation.generalized_dice import GeneralizedDiceScore
from unittests import NUM_CLASSES
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester
from unittests.segmentation.inputs import _input4, _inputs1, _inputs2, _inputs3

seed_all(42)


def _reference_generalized_dice(
    preds: torch.Tensor,
    target: torch.Tensor,
    input_format: str,
    include_background: bool = True,
    reduce: bool = True,
):
    """Calculate reference metric for generalized dice metric."""
    if input_format == "index":
        preds = torch.nn.functional.one_hot(preds, num_classes=NUM_CLASSES).movedim(-1, 1)
        target = torch.nn.functional.one_hot(target, num_classes=NUM_CLASSES).movedim(-1, 1)
    monai_extra_arg = {"sum_over_classes": True} if RequirementCache("monai>=1.4.0") else {}
    val = compute_generalized_dice(preds, target, include_background=include_background, **monai_extra_arg)
    if reduce:
        val = val.mean()
    return val.squeeze()


@pytest.mark.parametrize(
    "preds, target, input_format",
    [
        (_inputs1.preds, _inputs1.target, "one-hot"),
        (_inputs2.preds, _inputs2.target, "one-hot"),
        (_inputs3.preds, _inputs3.target, "index"),
        (_input4.preds, _input4.target, "index"),
    ],
)
@pytest.mark.parametrize("include_background", [True, False])
class TestGeneralizedDiceScore(MetricTester):
    """Test class for `GeneralizedDiceScore` metric."""

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
