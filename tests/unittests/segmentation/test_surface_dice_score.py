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

from torchmetrics.functional.segmentation.surface_dice import surface_dice_score
from torchmetrics.segmentation.surface_dice import SurfaceDiceScore
from unittests import NUM_BATCHES, _Input
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

seed_all(42)

if RequirementCache("monai>=1.4.0"):
    from monai.metrics.surface_dice import compute_surface_dice as monai_surface_dice
else:
    monai_surface_dice = None

NUM_CLASSES = 3
BATCH_SIZE = 4

to_one_hot = lambda x: torch.nn.functional.one_hot(x, NUM_CLASSES).movedim(-1, 2)

_one_hot_inputs = _Input(
    preds=to_one_hot(torch.randint(0, NUM_CLASSES, (NUM_BATCHES, BATCH_SIZE, 16, 16))),
    target=to_one_hot(torch.randint(0, NUM_CLASSES, (NUM_BATCHES, BATCH_SIZE, 16, 16))),
)
_index_inputs = _Input(
    preds=torch.randint(0, NUM_CLASSES, (NUM_BATCHES, BATCH_SIZE, 16, 16)),
    target=torch.randint(0, NUM_CLASSES, (NUM_BATCHES, BATCH_SIZE, 16, 16)),
)
_mixed_inputs = _Input(
    preds=(torch.rand((NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, 16, 16)) * 12 - 6),
    target=torch.randint(0, NUM_CLASSES, (NUM_BATCHES, BATCH_SIZE, 16, 16)),
)
_one_hot_3d_inputs = _Input(
    preds=to_one_hot(torch.randint(0, NUM_CLASSES, (NUM_BATCHES, BATCH_SIZE, 8, 8, 8))),
    target=to_one_hot(torch.randint(0, NUM_CLASSES, (NUM_BATCHES, BATCH_SIZE, 8, 8, 8))),
)


def _reference_surface_dice(preds, target, input_format, class_thresholds, include_background, spacing, reduce):
    """Reference implementation of surface Dice using MONAI."""
    if monai_surface_dice is None:
        raise RuntimeError("MONAI reference requested but not installed.")

    if input_format == "index":
        preds = torch.nn.functional.one_hot(preds, num_classes=NUM_CLASSES).movedim(-1, 1)
        target = torch.nn.functional.one_hot(target, num_classes=NUM_CLASSES).movedim(-1, 1)
    elif input_format == "mixed":
        preds = preds.argmax(dim=1)
        preds = torch.nn.functional.one_hot(preds, num_classes=NUM_CLASSES).movedim(-1, 1)
        target = torch.nn.functional.one_hot(target, num_classes=NUM_CLASSES).movedim(-1, 1)

    score = monai_surface_dice(
        preds.float(),
        target.float(),
        class_thresholds=class_thresholds,
        include_background=include_background,
        spacing=spacing,
        use_subvoxels=True,
    )
    if reduce:
        valid = ~score.isnan()
        score = torch.where(
            valid.any(dim=-1),
            torch.nan_to_num(score, nan=0.0).sum(dim=-1) / valid.sum(dim=-1),
            torch.nan,
        )
        return torch.nanmean(score)
    return score


@pytest.mark.parametrize(
    ("input_format", "inputs"),
    [("one-hot", _one_hot_inputs), ("index", _index_inputs), ("mixed", _mixed_inputs)],
)
@pytest.mark.parametrize("include_background", [True, False])
@pytest.mark.parametrize("spacing", [None, [1.0, 1.0], [1.5, 0.5]])
@pytest.mark.skipif(not RequirementCache("monai>=1.4.0"), reason="This test requires monai>=1.4.0 to be installed.")
class TestSurfaceDiceScore(MetricTester):
    """Test class for `SurfaceDiceScore` metric."""

    atol = 1e-5

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_surface_dice_score_class(self, input_format, inputs, include_background, spacing, ddp):
        """Test class implementation of metric."""
        preds, target = inputs
        thresholds = [1.0, 1.5, 2.0]
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=SurfaceDiceScore,
            reference_metric=partial(
                _reference_surface_dice,
                input_format=input_format,
                class_thresholds=thresholds,
                include_background=include_background,
                spacing=spacing,
                reduce=True,
            ),
            metric_args={
                "num_classes": NUM_CLASSES,
                "class_thresholds": thresholds if include_background else thresholds[1:],
                "include_background": include_background,
                "spacing": spacing,
                "input_format": input_format,
            },
        )

    def test_surface_dice_score_functional(self, input_format, inputs, include_background, spacing):
        """Test functional implementation of metric."""
        preds, target = inputs
        thresholds = [1.0, 1.5, 2.0]
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=surface_dice_score,
            reference_metric=partial(
                _reference_surface_dice,
                input_format=input_format,
                class_thresholds=thresholds,
                include_background=include_background,
                spacing=spacing,
                reduce=False,
            ),
            metric_args={
                "num_classes": NUM_CLASSES,
                "class_thresholds": thresholds if include_background else thresholds[1:],
                "include_background": include_background,
                "spacing": spacing,
                "per_class": True,
                "input_format": input_format,
            },
        )


@pytest.mark.parametrize("spacing", [None, [1.0, 1.0, 1.0], [1.0, 1.5, 2.0]])
@pytest.mark.skipif(not RequirementCache("monai>=1.4.0"), reason="This test requires monai>=1.4.0 to be installed.")
def test_surface_dice_score_functional_3d_reference(spacing):
    """Test 3D functional implementation against MONAI."""
    preds, target = _one_hot_3d_inputs
    expected = _reference_surface_dice(
        preds=preds,
        target=target,
        input_format="one-hot",
        class_thresholds=[1.0, 1.5, 2.0],
        include_background=True,
        spacing=spacing,
        reduce=False,
    )
    actual = surface_dice_score(
        preds=preds,
        target=target,
        num_classes=NUM_CLASSES,
        class_thresholds=[1.0, 1.5, 2.0],
        include_background=True,
        spacing=spacing,
        per_class=True,
    )
    assert torch.allclose(actual, expected, atol=1e-5, equal_nan=True)


def test_surface_dice_score_perfect_match():
    """A perfect surface match should have score 1."""
    preds = torch.zeros(1, 8, 8, dtype=torch.long)
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    preds[:, 2:6, 2:6] = 1
    target[:, 2:6, 2:6] = 1
    score = surface_dice_score(preds, target, num_classes=2, class_thresholds=1.0, input_format="index")
    assert torch.allclose(score, torch.ones_like(score))


def test_surface_dice_score_tolerance_sensitivity():
    """Shifted boundaries should be recovered once the tolerance is large enough."""
    preds = torch.zeros(1, 8, 8, dtype=torch.long)
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    preds[:, 2:6, 2:6] = 1
    target[:, 2:6, 3:7] = 1

    strict = surface_dice_score(preds, target, num_classes=2, class_thresholds=0.0, input_format="index")
    tolerant = surface_dice_score(preds, target, num_classes=2, class_thresholds=1.0, input_format="index")

    assert strict.item() < tolerant.item()
    assert torch.allclose(tolerant, torch.ones_like(tolerant))


def test_surface_dice_score_empty_and_one_sided_cases():
    """Absent classes should be ignored while one-sided surfaces score 0."""
    preds = torch.zeros(1, 8, 8, dtype=torch.long)
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    preds[:, 2:6, 2:6] = 1

    per_class = surface_dice_score(
        preds, target, num_classes=3, class_thresholds=[1.0, 1.0], per_class=True, input_format="index"
    )
    reduced = surface_dice_score(
        preds, target, num_classes=3, class_thresholds=[1.0, 1.0], per_class=False, input_format="index"
    )

    assert per_class.shape == (1, 2)
    assert per_class[0, 0].item() == 0.0
    assert torch.isnan(per_class[0, 1])
    assert reduced.item() == 0.0


def test_surface_dice_score_class_metric_matches_functional():
    """The class metric should match the functional result on simple data."""
    preds = torch.zeros(2, 2, 8, 8, dtype=torch.int)
    target = torch.zeros(2, 2, 8, 8, dtype=torch.int)
    preds[:, 1, 2:6, 2:6] = 1
    target[:, 1, 2:6, 3:7] = 1

    metric = SurfaceDiceScore(num_classes=2, class_thresholds=1.0)
    actual = metric(preds, target)
    expected = surface_dice_score(preds, target, num_classes=2, class_thresholds=1.0).mean()
    assert torch.allclose(actual, expected)


def test_surface_dice_score_raises_error():
    """Check that metric raises appropriate errors."""
    preds = torch.zeros(1, 8, 8, dtype=torch.long)
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    with pytest.raises(ValueError, match="positive integer"):
        surface_dice_score(preds, target, num_classes=0, class_thresholds=1.0, input_format="index")
    with pytest.raises(ValueError, match="one threshold per evaluated class"):
        surface_dice_score(preds, target, num_classes=NUM_CLASSES, class_thresholds=[1.0], input_format="index")
    with pytest.raises(ValueError, match="finite and non-negative"):
        surface_dice_score(preds, target, num_classes=NUM_CLASSES, class_thresholds=[1.0, -1.0], input_format="index")
    with pytest.raises(ValueError, match="length 2"):
        surface_dice_score(
            preds, target, num_classes=NUM_CLASSES, class_thresholds=[1.0, 1.0], spacing=[1.0], input_format="index"
        )
