# Copyright The Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
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
                # Preds is logits/probs: argmax then one-hot
                preds = preds.argmax(dim=1)
                preds = torch.nn.functional.one_hot(preds, num_classes=NUM_CLASSES).movedim(-1, 1)
            # else: preds is already one-hot integer tensor, no conversion needed
            target = torch.nn.functional.one_hot(target, num_classes=NUM_CLASSES).movedim(-1, 1)
        elif (preds.dim() + 1) == target.dim():
            if torch.is_floating_point(target):
                target = target.argmax(dim=1)
                target = torch.nn.functional.one_hot(target, num_classes=NUM_CLASSES).movedim(-1, 1)
            # else: target is already one-hot integer tensor, no conversion needed
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


class TestGeneralizedDiceScoreAbsentClasses:
    """Test that absent classes are handled correctly with nan (regression test for issue #2846).

    When a class is absent from a sample (both prediction and target are zero), the
    per-sample score for that class should be ``nan``. When averaging across samples,
    ``nan`` values should be excluded so that absent-class samples do not drag the
    per-class score down incorrectly.

    """

    @staticmethod
    def _make_absent_class_data(num_classes=3, spatial=128):
        """Create data where some classes are absent from certain samples.

        - Sample 0: class 0 present, perfectly predicted
        - Sample 2: class 1 present, perfectly predicted
        - Samples 1, 3: no class present (all zeros)

        """
        target = torch.zeros(4, num_classes, spatial, spatial, dtype=torch.int8)
        preds = torch.zeros(4, num_classes, spatial, spatial, dtype=torch.int8)
        target[0, 0] = 1
        preds[0, 0] = 1
        target[2, 1] = 1
        preds[2, 1] = 1
        return preds, target

    def test_functional_per_class_absent(self):
        """Per-sample per-class scores should be nan for absent classes."""
        preds, target = self._make_absent_class_data()
        result = generalized_dice_score(preds, target, num_classes=3, per_class=True, include_background=True)
        # Sample 0: class 0 present -> score 1.0; classes 1,2 absent -> nan
        assert result[0, 0] == 1.0
        assert torch.isnan(result[0, 1])
        assert torch.isnan(result[0, 2])
        # Sample 1: all classes absent -> nan
        assert torch.isnan(result[1]).all()
        # Sample 2: class 1 present -> score 1.0; classes 0,2 absent -> nan
        assert torch.isnan(result[2, 0])
        assert result[2, 1] == 1.0
        assert torch.isnan(result[2, 2])
        # Sample 3: all classes absent -> nan
        assert torch.isnan(result[3]).all()

    def test_functional_per_class_false_absent(self):
        """Per_class=False should sum across classes before dividing, giving nan only for samples where ALL classes are
        absent.
        """
        preds, target = self._make_absent_class_data()
        result = generalized_dice_score(preds, target, num_classes=3, per_class=False, include_background=True)
        # Samples 0 and 2 have at least one class present -> score should be 1.0
        assert result[0] == 1.0
        assert result[2] == 1.0
        # Samples 1 and 3 have no classes present -> nan
        assert torch.isnan(result[1])
        assert torch.isnan(result[3])

    def test_class_per_class_absent(self):
        """Class metric with per_class=True should return nan for classes absent from all samples, and exclude absent-
        class samples from the average.
        """
        preds, target = self._make_absent_class_data()
        gds = GeneralizedDiceScore(num_classes=3, per_class=True, include_background=True)
        result = gds(preds, target)
        # Class 0: only sample 0 has it present, score 1.0 -> average 1.0
        assert result[0] == 1.0
        # Class 1: only sample 2 has it present, score 1.0 -> average 1.0
        assert result[1] == 1.0
        # Class 2: absent from all samples -> nan
        assert torch.isnan(result[2])

    def test_class_per_class_false_absent(self):
        """Class metric with per_class=False should work correctly even with absent classes."""
        preds, target = self._make_absent_class_data()
        gds = GeneralizedDiceScore(num_classes=3, per_class=False, include_background=True)
        result = gds(preds, target)
        # Average over non-nan sample scores (1.0, 1.0) = 1.0
        assert result == 1.0

    def test_class_multiple_updates_absent(self):
        """Test that multiple update calls correctly handle absent classes."""
        gds = GeneralizedDiceScore(num_classes=3, per_class=True, include_background=True)

        # Batch 1: class 0 present and correct
        preds1 = torch.zeros(2, 3, 4, 4, dtype=torch.int8)
        target1 = torch.zeros(2, 3, 4, 4, dtype=torch.int8)
        target1[0, 0] = 1
        preds1[0, 0] = 1
        target1[1, 1] = 1
        preds1[1, 1] = 1

        # Batch 2: class 0 partially correct (1 correct, 1 wrong)
        preds2 = torch.zeros(2, 3, 4, 4, dtype=torch.int8)
        target2 = torch.zeros(2, 3, 4, 4, dtype=torch.int8)
        target2[0, 0] = 1
        preds2[0, 0] = 1  # correct
        target2[1, 0] = 1  # present but prediction wrong (all zeros)

        gds.update(preds1, target1)
        gds.update(preds2, target2)
        result = gds.compute()

        # Class 0: 3 samples with class present, scores [1.0, 1.0, 0.0] -> mean = 2/3
        assert torch.isclose(result[0], torch.tensor(2.0 / 3.0), atol=1e-4)
        # Class 1: 1 sample present, score 1.0
        assert result[1] == 1.0
        # Class 2: absent from all -> nan
        assert torch.isnan(result[2])

    @pytest.mark.parametrize("weight_type", ["square", "simple", "linear"])
    def test_absent_classes_all_weight_types(self, weight_type):
        """Absent-class handling should work with all weight types."""
        preds, target = self._make_absent_class_data()
        result = generalized_dice_score(
            preds, target, num_classes=3, per_class=True, include_background=True, weight_type=weight_type
        )
        # Present classes should score 1.0 regardless of weight type
        assert result[0, 0] == 1.0
        assert result[2, 1] == 1.0
        # Absent classes should be nan
        assert torch.isnan(result[0, 1])
        assert torch.isnan(result[0, 2])
        assert torch.isnan(result[2, 0])
        assert torch.isnan(result[2, 2])
