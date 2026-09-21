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

import numpy as np
import pytest
import torch
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure

from torchmetrics.functional.segmentation.boundary_f_score import boundary_f_score
from torchmetrics.segmentation.boundary_f_score import BoundaryFScore
from unittests import NUM_CLASSES
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester
from unittests.segmentation.inputs import (
    _index_input_1,
    _mixed_input_1,
    _mixed_input_2,
    _mixed_logits_input,
    _one_hot_input_1,
    _one_hot_input_2,
)

seed_all(42)


def _format_reference_inputs(
    preds: torch.Tensor,
    target: torch.Tensor,
    input_format: str,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Format predictions and targets to one-hot tensors for the reference implementation."""
    if input_format == "one-hot":
        if torch.is_floating_point(preds):
            preds = preds.argmax(dim=1)
            preds = torch.nn.functional.one_hot(preds, num_classes=num_classes).movedim(-1, 1)
        if torch.is_floating_point(target):
            target = target.argmax(dim=1)
            target = torch.nn.functional.one_hot(target, num_classes=num_classes).movedim(-1, 1)
        return preds, target

    if input_format == "index":
        preds = torch.nn.functional.one_hot(preds, num_classes=num_classes).movedim(-1, 1)
        target = torch.nn.functional.one_hot(target, num_classes=num_classes).movedim(-1, 1)
        return preds, target

    if preds.dim() == (target.dim() + 1):
        if torch.is_floating_point(preds):
            preds = preds.argmax(dim=1)
            preds = torch.nn.functional.one_hot(preds, num_classes=num_classes).movedim(-1, 1)
        target = torch.nn.functional.one_hot(target, num_classes=num_classes).movedim(-1, 1)
        return preds, target

    if torch.is_floating_point(target):
        target = target.argmax(dim=1)
        target = torch.nn.functional.one_hot(target, num_classes=num_classes).movedim(-1, 1)
    preds = torch.nn.functional.one_hot(preds, num_classes=num_classes).movedim(-1, 1)
    return preds, target


def _reference_boundary_score_for_mask(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    boundary_width: int,
) -> float:
    """Reference boundary F-score for a single binary mask pair."""
    structure = generate_binary_structure(pred_mask.ndim, 1)
    pred_boundary = pred_mask & ~binary_erosion(pred_mask, structure=structure)
    target_boundary = target_mask & ~binary_erosion(target_mask, structure=structure)

    pred_boundary_area = pred_boundary.sum()
    target_boundary_area = target_boundary.sum()
    if pred_boundary_area == 0 and target_boundary_area == 0:
        return float("nan")

    if boundary_width > 0:
        dilation_structure = np.ones((2 * boundary_width + 1,) * pred_mask.ndim, dtype=bool)
        pred_neighborhood = binary_dilation(pred_boundary, structure=dilation_structure)
        target_neighborhood = binary_dilation(target_boundary, structure=dilation_structure)
    else:
        pred_neighborhood = pred_boundary
        target_neighborhood = target_boundary

    matched_pred = np.logical_and(pred_boundary, target_neighborhood).sum()
    matched_target = np.logical_and(target_boundary, pred_neighborhood).sum()

    precision = matched_pred / pred_boundary_area if pred_boundary_area > 0 else 0.0
    recall = matched_target / target_boundary_area if target_boundary_area > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _reference_boundary_f_score(
    preds: torch.Tensor,
    target: torch.Tensor,
    input_format: str,
    num_classes: int,
    include_background: bool = True,
    per_class: bool = True,
    boundary_width: int = 1,
    reduce: bool = True,
) -> torch.Tensor:
    """Reference implementation of the boundary F-score."""
    preds, target = _format_reference_inputs(preds, target, input_format, num_classes)
    if not include_background:
        preds = preds[:, 1:]
        target = target[:, 1:]

    preds = preds.bool().cpu().numpy()
    target = target.bool().cpu().numpy()

    scores = torch.tensor(
        [
            [
                _reference_boundary_score_for_mask(pred_mask, target_mask, boundary_width)
                for pred_mask, target_mask in zip(pred_sample, target_sample)
            ]
            for pred_sample, target_sample in zip(preds, target)
        ],
        dtype=torch.float32,
    )

    if not per_class:
        scores = torch.nanmean(scores, dim=-1)

    return torch.nanmean(scores, dim=0) if reduce else scores


@pytest.mark.parametrize(
    ("preds", "target", "input_format"),
    [
        (_one_hot_input_1.preds, _one_hot_input_1.target, "one-hot"),
        (_one_hot_input_2.preds, _one_hot_input_2.target, "one-hot"),
        (_index_input_1.preds, _index_input_1.target, "index"),
        (_mixed_input_1.preds, _mixed_input_1.target, "mixed"),
        (_mixed_input_2.preds, _mixed_input_2.target, "mixed"),
        (_mixed_logits_input.preds, _mixed_logits_input.target, "mixed"),
    ],
)
@pytest.mark.parametrize("include_background", [True, False])
@pytest.mark.parametrize("per_class", [True, False])
@pytest.mark.parametrize("boundary_width", [0, 1])
class TestBoundaryFScore(MetricTester):
    """Test class for `BoundaryFScore` metric."""

    atol = 1e-4

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_boundary_f_score_class(
        self, preds, target, input_format, include_background, per_class, boundary_width, ddp
    ):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=BoundaryFScore,
            reference_metric=partial(
                _reference_boundary_f_score,
                input_format=input_format,
                num_classes=NUM_CLASSES,
                include_background=include_background,
                per_class=per_class,
                boundary_width=boundary_width,
                reduce=True,
            ),
            metric_args={
                "num_classes": NUM_CLASSES,
                "include_background": include_background,
                "per_class": per_class,
                "boundary_width": boundary_width,
                "input_format": input_format,
            },
        )

    def test_boundary_f_score_functional(
        self, preds, target, input_format, include_background, per_class, boundary_width
    ):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=boundary_f_score,
            reference_metric=partial(
                _reference_boundary_f_score,
                input_format=input_format,
                num_classes=NUM_CLASSES,
                include_background=include_background,
                per_class=per_class,
                boundary_width=boundary_width,
                reduce=False,
            ),
            metric_args={
                "num_classes": NUM_CLASSES,
                "include_background": include_background,
                "per_class": per_class,
                "boundary_width": boundary_width,
                "input_format": input_format,
            },
        )


def test_boundary_f_score_perfect_match() -> None:
    """Test that a perfect boundary match gives a score of one."""
    preds = torch.zeros(1, 2, 8, 8, dtype=torch.int)
    target = torch.zeros(1, 2, 8, 8, dtype=torch.int)
    preds[:, 1, 2:6, 2:6] = 1
    target[:, 1, 2:6, 2:6] = 1
    score = boundary_f_score(preds, target, num_classes=2, include_background=False, per_class=True)
    assert torch.allclose(score, torch.tensor([[1.0]]))


def test_boundary_f_score_mismatch() -> None:
    """Test that completely mismatched non-empty boundaries score zero."""
    preds = torch.zeros(1, 2, 8, 8, dtype=torch.int)
    target = torch.zeros(1, 2, 8, 8, dtype=torch.int)
    preds[:, 1, :3, :3] = 1
    target[:, 1, 5:, 5:] = 1
    score = boundary_f_score(preds, target, num_classes=2, include_background=False)
    assert score.item() == 0.0


def test_boundary_f_score_respects_tolerance() -> None:
    """Test that larger tolerance accepts slightly shifted boundaries."""
    preds = torch.zeros(1, 2, 10, 10, dtype=torch.int)
    target = torch.zeros(1, 2, 10, 10, dtype=torch.int)
    preds[:, 1, 2:6, 2:6] = 1
    target[:, 1, 2:6, 3:7] = 1

    small_tolerance = boundary_f_score(preds, target, num_classes=2, include_background=False, boundary_width=0)
    large_tolerance = boundary_f_score(preds, target, num_classes=2, include_background=False, boundary_width=1)

    assert small_tolerance.item() < 1.0
    assert large_tolerance.item() == 1.0


@pytest.mark.parametrize(
    ("preds", "target", "expected"),
    [
        (torch.zeros(1, 2, 8, 8, dtype=torch.int), torch.zeros(1, 2, 8, 8, dtype=torch.int), float("nan")),
        (
            torch.zeros(1, 2, 8, 8, dtype=torch.int),
            torch.nn.functional.one_hot(torch.ones(1, 8, 8, dtype=torch.long), num_classes=2).movedim(-1, 1),
            0.0,
        ),
        (
            torch.nn.functional.one_hot(torch.ones(1, 8, 8, dtype=torch.long), num_classes=2).movedim(-1, 1),
            torch.zeros(1, 2, 8, 8, dtype=torch.int),
            0.0,
        ),
    ],
)
def test_boundary_f_score_empty_boundary_cases(preds: torch.Tensor, target: torch.Tensor, expected: float) -> None:
    """Test empty-boundary behavior."""
    score = boundary_f_score(preds, target, num_classes=2, include_background=False)
    expected_tensor = torch.tensor(expected)
    assert torch.allclose(score, expected_tensor, equal_nan=True)


def test_boundary_f_score_supports_3d_inputs() -> None:
    """Test that the metric supports volumetric masks."""
    preds = torch.zeros(1, 2, 6, 6, 6, dtype=torch.int)
    target = torch.zeros(1, 2, 6, 6, 6, dtype=torch.int)
    preds[:, 1, 1:5, 1:5, 1:5] = 1
    target[:, 1, 1:5, 1:5, 1:5] = 1
    metric = BoundaryFScore(num_classes=2, include_background=False)
    assert metric(preds, target).item() == 1.0


def test_boundary_f_score_invalid_args_and_shapes() -> None:
    """Test argument and shape validation."""
    with pytest.raises(ValueError, match="Expected argument `boundary_width` to be a non-negative integer"):
        BoundaryFScore(num_classes=2, boundary_width=-1)

    preds = torch.randint(0, 2, (2, 3, 8), dtype=torch.int)
    target = torch.randint(0, 2, (2, 3, 8), dtype=torch.int)
    with pytest.raises(ValueError, match="Expected `preds` and `target` to have 2D or 3D spatial dimensions"):
        boundary_f_score(preds, target, num_classes=3)
