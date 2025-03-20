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
from sklearn.metrics import f1_score

from torchmetrics import MetricCollection
from torchmetrics.functional.segmentation.dice import dice_score
from torchmetrics.segmentation.dice import DiceScore
from unittests import NUM_CLASSES
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester
from unittests.segmentation.inputs import _inputs1, _inputs2, _inputs3, _inputs4

seed_all(42)


def _reference_dice_score(
    preds: torch.Tensor,
    target: torch.Tensor,
    input_format: str,
    include_background: bool = True,
    average: str = "micro",
    aggregation_level: str = "samplewise",
):
    """Calculate reference metric for dice score."""
    if input_format == "one-hot":
        preds = preds.argmax(dim=1)
        target = target.argmax(dim=1)
    preds = preds.cpu().numpy()
    target = target.cpu().numpy()

    labels = list(range(1, NUM_CLASSES) if not include_background else range(NUM_CLASSES))
    if aggregation_level == "samplewise":
        val = [f1_score(t.flatten(), p.flatten(), average=average, labels=labels) for t, p in zip(target, preds)]
        return torch.tensor(val).mean(dim=0)
    if aggregation_level == "global":
        val = f1_score(target.flatten(), preds.flatten(), average=average, labels=labels)
        return torch.tensor(val)
    raise ValueError(f"Unknown aggregation level: {aggregation_level}.")


@pytest.mark.parametrize(
    "preds, target, input_format",
    [
        (_inputs1.preds, _inputs1.target, "one-hot"),
        (_inputs2.preds, _inputs2.target, "one-hot"),
        (_inputs3.preds, _inputs3.target, "index"),
        (_inputs4.preds, _inputs4.target, "index"),
    ],
)
@pytest.mark.parametrize("include_background", [True, False])
@pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
@pytest.mark.parametrize("aggregation_level", ["samplewise", "global"])
class TestDiceScore(MetricTester):
    """Test class for `DiceScore` metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_dice_score_class(self, preds, target, input_format, include_background, average, aggregation_level, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=DiceScore,
            reference_metric=partial(
                _reference_dice_score,
                input_format=input_format,
                include_background=include_background,
                average=average,
                aggregation_level=aggregation_level,
            ),
            metric_args={
                "num_classes": NUM_CLASSES,
                "include_background": include_background,
                "average": average,
                "input_format": input_format,
                "aggregation_level": aggregation_level,
            },
        )

    def test_dice_score_functional(self, preds, target, input_format, include_background, average, aggregation_level):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=dice_score,
            reference_metric=partial(
                _reference_dice_score,
                input_format=input_format,
                include_background=include_background,
                average=average,
                aggregation_level=aggregation_level,
            ),
            metric_args={
                "num_classes": NUM_CLASSES,
                "include_background": include_background,
                "average": average,
                "input_format": input_format,
                "aggregation_level": aggregation_level,
                "reduce": True,
            },
        )


@pytest.mark.parametrize(
    ("average", "aggregation_level", "expected_score"),
    [
        ("micro", "samplewise", 0.2),
        ("macro", "samplewise", 0.2),
        ("weighted", "samplewise", 0.1333),
        ("none", "samplewise", [torch.nan, 0.4, torch.nan, 0.2]),
        ("micro", "global", 0.3768),
        ("macro", "global", 0.3667),
        ("weighted", "global", 0.3795),
        ("none", "global", [torch.nan, 0.4, torch.nan, 0.3333]),
    ],
)
def test_samples_with_missing_classes(average, aggregation_level, expected_score):
    """Tests case where not all classes are present in all samples."""
    num_samples, num_classes = 3, 4
    target = torch.full((num_samples, num_classes, 10, 10), 0, dtype=torch.int8)
    preds = torch.full((num_samples, num_classes, 10, 10), 0, dtype=torch.int8)

    # sample with class 1 & 3
    target[0, 1, :6, :6], preds[0, 1, :3, :3] = 1, 1
    target[0, 3, 6:, 6:], preds[0, 3, 8:, 8:] = 1, 1

    # sample with only background and false positives
    preds[1, 3, :2, :2] = 1

    dice = DiceScore(
        num_classes=num_classes, average=average, include_background=True, aggregation_level=aggregation_level
    )
    score = dice(preds, target)
    assert torch.allclose(score, torch.tensor(expected_score), equal_nan=True, atol=1e-4)


@pytest.mark.parametrize("average", ["micro", None])
@pytest.mark.parametrize("aggregation_level", ["samplewise", "global"])
def test_corner_case_zero_denominator(aggregation_level, average):
    """Check that the metric returns NaN when the denominator is all zero."""
    num_classes = 3
    target = torch.full((4, num_classes, 128, 128), 0, dtype=torch.int8)
    preds = torch.full((4, num_classes, 128, 128), 0, dtype=torch.int8)
    dice = DiceScore(
        num_classes=num_classes, average=average, include_background=True, aggregation_level=aggregation_level
    )
    score = dice(preds, target)
    if average is None:
        assert len(score) == num_classes
        assert all(t.isnan() for t in score)
    else:
        assert score.isnan()


@pytest.mark.parametrize("compute_groups", [True, False])
def test_dice_score_metric_collection(compute_groups: bool, num_batches: int = 4):
    """Test that the metric works within a metric collection with and without compute groups."""
    metric_collection = MetricCollection(
        metrics={
            "DiceScore (micro)": DiceScore(
                num_classes=NUM_CLASSES,
                average="micro",
            ),
            "DiceScore (macro)": DiceScore(
                num_classes=NUM_CLASSES,
                average="macro",
            ),
            "DiceScore (weighted)": DiceScore(
                num_classes=NUM_CLASSES,
                average="weighted",
            ),
        },
        compute_groups=compute_groups,
    )

    for _ in range(num_batches):
        metric_collection.update(_inputs1.preds, _inputs1.target)
    result = metric_collection.compute()

    assert isinstance(result, dict)
    assert len(set(metric_collection.keys()) - set(result.keys())) == 0
