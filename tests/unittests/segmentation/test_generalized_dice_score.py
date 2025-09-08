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
from unittests.segmentation.inputs import (
    _index_input_1,
    _index_input_2,
    _mixed_input_1,
    _mixed_input_2,
    _mixed_logits_input,
    _one_hot_input_1,
    _one_hot_input_2,
)

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
    elif input_format == "mixed":
        if preds.dim() == (target.dim() + 1):
            if torch.is_floating_point(preds):
                preds = preds.argmax(dim=1)
                preds = torch.nn.functional.one_hot(preds, num_classes=NUM_CLASSES).movedim(-1, 1)
            target = torch.nn.functional.one_hot(target, num_classes=NUM_CLASSES).movedim(-1, 1)
        elif (preds.dim() + 1) == target.dim():
            if torch.is_floating_point(target):
                target = target.argmax(dim=1)
                target = torch.nn.functional.one_hot(target, num_classes=NUM_CLASSES).movedim(-1, 1)
            preds = torch.nn.functional.one_hot(preds, num_classes=NUM_CLASSES).movedim(-1, 1)
    monai_extra_arg = {"sum_over_classes": True} if RequirementCache("monai>=1.4.0") else {}
    val = compute_generalized_dice(preds, target, include_background=include_background, **monai_extra_arg)
    if reduce:
        val = val.mean()
    return val.squeeze()


@pytest.mark.parametrize(
    ("preds", "target", "input_format"),
    [
        (_one_hot_input_1.preds, _one_hot_input_1.target, "one-hot"),
        (_one_hot_input_2.preds, _one_hot_input_2.target, "one-hot"),
        (_index_input_1.preds, _index_input_1.target, "index"),
        (_index_input_2.preds, _index_input_2.target, "index"),
        (_mixed_input_1.preds, _mixed_input_1.target, "mixed"),
        (_mixed_input_2.preds, _mixed_input_2.target, "mixed"),
        (_mixed_logits_input.preds, _mixed_logits_input.target, "mixed"),
    ],
)
@pytest.mark.parametrize("include_background", [True, False])
class TestGeneralizedDiceScore(MetricTester):
    """Test class for `GeneralizedDiceScore` metric."""

    atol = 1e-4

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


@pytest.mark.parametrize("per_class", [True, False])
@pytest.mark.parametrize("include_background", [True, False])
def test_samples_with_missing_classes(per_class, include_background):
    """Test GeneralizedDiceScore with missing classes in some samples."""
    target = torch.zeros((4, NUM_CLASSES, 128, 128), dtype=torch.int8)
    preds = torch.zeros((4, NUM_CLASSES, 128, 128), dtype=torch.int8)

    target[0, 0, 0, 0] = 1
    preds[0, 0, 0, 0] = 1

    target[2, 1, 0, 0] = 1
    preds[2, 1, 0, 0] = 1

    metric = GeneralizedDiceScore(num_classes=NUM_CLASSES, per_class=per_class, include_background=include_background)
    score = metric(preds, target)

    target_slice = target if include_background else target[:, 1:]
    output_classes = NUM_CLASSES if include_background else NUM_CLASSES - 1

    if per_class:
        assert len(score) == output_classes
        for c in range(output_classes):
            assert score[c] == 1.0 if target_slice[:, c].sum() > 0 else torch.isnan(score[c])
    else:
        assert score.isnan()


@pytest.mark.parametrize("per_class", [True, False])
@pytest.mark.parametrize("include_background", [True, False])
def test_generalized_dice_zero_denominator(per_class, include_background):
    """Check that GeneralizedDiceScore returns NaN when the denominator is all zero (no class present)."""
    target = torch.full((4, NUM_CLASSES, 128, 128), 0, dtype=torch.int8)
    preds = torch.full((4, NUM_CLASSES, 128, 128), 0, dtype=torch.int8)

    metric = GeneralizedDiceScore(num_classes=NUM_CLASSES, per_class=per_class, include_background=include_background)

    score = metric(preds, target)

    if per_class and include_background:
        assert len(score) == NUM_CLASSES
        assert all(t.isnan() for t in score)
    elif per_class and not include_background:
        assert len(score) == NUM_CLASSES - 1
        assert all(t.isnan() for t in score)
    else:
        # Expect scalar NaN
        assert score.isnan()
