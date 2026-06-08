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
    def _make_absent_class_data(num_classes=3, spatial=128) -> tuple[torch.Tensor, torch.Tensor]:
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
        """Per_class=False should sum across classes before dividing.

        Gives nan only for samples where ALL classes are absent.

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
        """Class metric with per_class=True should return nan for classes absent from all samples.

        Excludes absent-class samples from the average.

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

    def test_functional_fp_on_absent_per_class_true(self):
        """FP-on-absent class should score 0.0 (not NaN) when the class appears in another sample.

        Regression test for F2: absent = support == 0 silently set FP-on-absent to NaN.

        The FP penalty requires the class to appear in at least one sample's target so that
        w_max > 0. With two samples where sample 1 has class 1 present, sample 0's FP on
        class 1 gets weight w_max > 0 and scores 0.0 instead of NaN.

        """
        preds = torch.zeros(2, 2, 4, 4, dtype=torch.int8)
        target = torch.zeros(2, 2, 4, 4, dtype=torch.int8)
        # Sample 0: class 0 perfect, class 1 FP-on-absent
        target[0, 0] = 1
        preds[0, 0] = 1
        preds[0, 1, :2] = 1  # class 1: 8 FP pixels
        # Sample 1: class 0 perfect, class 1 present and perfect (w_max for class 1 > 0)
        target[1, 0] = 1
        preds[1, 0] = 1
        target[1, 1] = 1
        preds[1, 1] = 1
        result = generalized_dice_score(preds, target, num_classes=2, per_class=True)
        assert result[0, 0] == 1.0
        assert not torch.isnan(result[0, 1]), "FP-on-absent must not be NaN when class appears in another sample"
        assert result[0, 1] == 0.0, "FP-on-absent should score 0.0"
        assert result[1, 0] == 1.0
        assert result[1, 1] == 1.0

    def test_functional_fp_on_absent_per_class_false(self):
        """FP-on-absent class must lower the aggregate score below 1.0.

        Regression test for F3: present mask dropped FP-on-absent denominator, inflating
        score to 1.0. With linear weights: (2N)/(2N+M) where N=M=1 -> 2/3 ~= 0.667.

        """
        preds = torch.zeros(1, 2, 1, 1, dtype=torch.int8)
        target = torch.zeros(1, 2, 1, 1, dtype=torch.int8)
        target[0, 0] = 1  # class 0: 1 pixel in target
        preds[0, 0] = 1  # class 0: correctly predicted
        preds[0, 1] = 1  # class 1: 1 FP pixel, absent from target
        result = generalized_dice_score(preds, target, num_classes=2, per_class=False, weight_type="linear")
        expected = torch.tensor(2.0 / 3.0)
        assert torch.isclose(result[0], expected, atol=1e-4), f"expected {expected:.4f}, got {result[0]:.4f}"
        assert result[0] < 1.0, "FP predictions must lower the aggregate score below 1.0"

    def test_functional_nontrivial_score_with_fp_on_absent(self):
        """Per_class=True: non-trivial present-class score coexists with FP-on-absent 0.0.

        Verifies that a partially-correct present class is not contaminated by NaN from the adjacent FP-on-absent class
        (F2 regression).

        """
        preds = torch.zeros(1, 2, 4, 1, dtype=torch.int8)
        target = torch.zeros(1, 2, 4, 1, dtype=torch.int8)
        target[0, 0, :4] = 1  # class 0: 4 pixels in target
        preds[0, 0, :3] = 1  # class 0: 3/4 correct (partial prediction)
        preds[0, 1, :2] = 1  # class 1: 2 FP pixels, absent from target
        result = generalized_dice_score(preds, target, num_classes=2, per_class=True, weight_type="linear")
        # class 0: intersection=3, target_sum=4, pred_sum=3 -> 2*3/(4+3) = 6/7
        expected_c0 = torch.tensor(6.0 / 7.0)
        assert torch.isclose(result[0, 0], expected_c0, atol=1e-4), (
            f"expected {expected_c0:.4f}, got {result[0, 0]:.4f}"
        )
        assert not torch.isnan(result[0, 0]), "partially-correct class must not be NaN"
        assert result[0, 1] == 0.0, "FP-on-absent must score 0.0"
        assert not torch.isnan(result[0, 1]), "FP-on-absent must not be NaN"

    def test_class_metric_fp_on_absent(self):
        """Module metric: FP-on-absent averages to 0.0 (not NaN) when class appears in another sample.

        Regression test for F4: module compute() inherits per-sample NaN from _generalized_dice_compute.
        Uses 2 samples so that w_max > 0 for the FP class (sample 1 has class 1 present).

        """
        preds = torch.zeros(2, 2, 4, 4, dtype=torch.int8)
        target = torch.zeros(2, 2, 4, 4, dtype=torch.int8)
        target[0, 0] = 1
        preds[0, 0] = 1
        preds[0, 1, :2] = 1  # sample 0: class 1 FP
        target[1, 0] = 1
        preds[1, 0] = 1
        target[1, 1] = 1
        preds[1, 1] = 1  # sample 1: class 1 present and perfect (w_max for class 1 > 0)
        gds = GeneralizedDiceScore(num_classes=2, per_class=True)
        result = gds(preds, target)
        assert result[0] == 1.0
        assert not torch.isnan(result[1]), "FP-on-absent class must not produce NaN in module compute()"
        # class 1: [0.0 (FP), 1.0 (present)] -> nanmean = 0.5
        assert torch.isclose(result[1], torch.tensor(0.5), atol=1e-4)

    def test_functional_fp_on_absent_include_background_false(self):
        """FP-on-absent handling is consistent when include_background=False.

        Regression test for F6e: combined include_background=False x absent-class path.
        Uses 2 samples so that w_max > 0 for the FP class.

        """
        preds = torch.zeros(2, 3, 4, 4, dtype=torch.int8)
        target = torch.zeros(2, 3, 4, 4, dtype=torch.int8)
        # Sample 0: class 1 perfect, class 2 FP-on-absent
        target[0, 1] = 1
        preds[0, 1] = 1
        preds[0, 2, :2] = 1  # class 2 FP
        # Sample 1: class 1 and class 2 both present and perfect (w_max for class 2 > 0)
        target[1, 1] = 1
        preds[1, 1] = 1
        target[1, 2] = 1
        preds[1, 2] = 1
        result = generalized_dice_score(preds, target, num_classes=3, per_class=True, include_background=False)
        # With include_background=False: class 0 excluded; columns map to [class 1, class 2]
        assert result.shape[1] == 2, "include_background=False should exclude background class"
        assert result[0, 0] == 1.0, "class 1 (non-background, present) should score 1.0"
        assert result[0, 1] == 0.0, "class 2 (non-background, FP-on-absent) should score 0.0"
        assert not torch.isnan(result[0, 1])
