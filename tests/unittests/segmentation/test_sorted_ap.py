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
"""Tests for the sortedAP metric.

The reference implementation used here is deliberately written from scratch (plain
python/numpy loops, independent matching code) rather than reusing any helper from
``torchmetrics.functional.segmentation.sorted_ap``, so that it acts as a genuine
test oracle instead of just re-checking the same code path.

"""

import importlib
from functools import partial
from typing import NamedTuple

import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from torchmetrics.functional.segmentation.sorted_ap import sorted_ap
from torchmetrics.segmentation.sorted_ap import SortedAP
from unittests import BATCH_SIZE, NUM_BATCHES
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

seed_all(42)


# --------------------------------------------------------------------------------------
# Independent reference implementation (test oracle)
# --------------------------------------------------------------------------------------


def _reference_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Plain-python IoU between two boolean masks."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection) / float(union) if union > 0 else float("nan")


def _reference_match(iou: np.ndarray, method: str) -> np.ndarray:
    """Independent matching implementation used as ground truth in tests."""
    match = np.zeros_like(iou, dtype=bool)
    cost = np.nan_to_num(iou, nan=0.0)
    cost[cost < 1e-6] = 0.0

    if method == "hungarian":
        idx_pd = cost.max(axis=1) > 0
        idx_gt = cost.max(axis=0) > 0
        if not idx_pd.any() or not idx_gt.any():
            return match
        sub_cost = cost[np.ix_(idx_pd, idx_gt)]
        row, col = linear_sum_assignment(1 - sub_cost)
        row = np.nonzero(idx_pd)[0][row]
        col = np.nonzero(idx_gt)[0][col]
        keep = cost[row, col] >= 1e-6
        match[row[keep], col[keep]] = True
    else:  # greedy
        work = cost.copy()
        while True:
            flat = int(np.argmax(work))
            r, c = np.unravel_index(flat, work.shape)
            if work[r, c] <= 0:
                break
            match[r, c] = True
            work[r, :] = 0.0
            work[:, c] = 0.0
    return match


def _reference_sorted_ap(preds: list[Tensor], target: list[Tensor], match_method: str = "hungarian") -> Tensor:
    """Fully independent re-implementation of Algorithm 2 from the sortedAP paper."""
    matched_ious: list[float] = []
    tp0 = fn0 = n_pd = 0

    for p, t in zip(preds, target):
        p_np = p.numpy().astype(bool)
        t_np = t.numpy().astype(bool)
        p_np = np.stack([m for m in p_np if m.any()]) if len(p_np) else p_np.reshape(0, *p_np.shape[1:])
        t_np = np.stack([m for m in t_np if m.any()]) if len(t_np) else t_np.reshape(0, *t_np.shape[1:])

        num_pd, num_gt = p_np.shape[0], t_np.shape[0]
        n_pd += num_pd
        if num_pd == 0 or num_gt == 0:
            fn0 += num_gt
            continue

        iou = np.full((num_pd, num_gt), np.nan)
        for i in range(num_pd):
            for j in range(num_gt):
                iou[i, j] = _reference_iou(p_np[i], t_np[j])

        match = _reference_match(iou, match_method)
        n_match = int(match.sum())
        tp0 += n_match
        fn0 += num_gt - n_match
        if n_match:
            matched_ious.extend(iou[match].tolist())

    if tp0 == 0 and fn0 != 0:
        return torch.tensor(0.0)
    if n_pd == 0 and fn0 == 0:
        return torch.tensor(1.0)
    if not matched_ious:
        return torch.tensor(0.0)

    ious = np.sort(np.array(matched_ious))
    ap0 = tp0 / (n_pd + fn0)
    score = ious[0] * ap0
    ap_prev, t_prev = ap0, ious[0]
    for k in range(1, tp0 + 1):
        ap_k = (tp0 - k) / (n_pd + fn0 + k)
        t_k = ious[k - 1]
        score += 0.5 * (t_k - t_prev) * (ap_k + ap_prev)
        ap_prev, t_prev = ap_k, t_k
    return torch.tensor(float(score))


# --------------------------------------------------------------------------------------
# Synthetic test inputs
# --------------------------------------------------------------------------------------


class _Input(NamedTuple):
    preds: list[list[Tensor]]
    target: list[list[Tensor]]


def _generate_image_masks(max_instances: int, height: int = 24, width: int = 24) -> tuple:
    """Generate one image's (preds, target) instance masks with partial, non-trivial overlap."""
    n_gt = int(torch.randint(1, max_instances + 1, (1,)))
    gt = torch.zeros(n_gt, height, width, dtype=torch.bool)
    for i in range(n_gt):
        h0 = int(torch.randint(0, height - 6, (1,)))
        w0 = int(torch.randint(0, width - 6, (1,)))
        gt[i, h0 : h0 + 6, w0 : w0 + 6] = True

    # predictions: a jittered/shifted version of (a subset of) the targets, so IoUs
    # land somewhere in (0, 1) rather than only at the degenerate 0/1 extremes.
    n_pd = max(0, n_gt + int(torch.randint(-1, 2, (1,))))
    pd = torch.zeros(n_pd, height, width, dtype=torch.bool)
    for i in range(n_pd):
        src = gt[i % n_gt].clone()
        shift = tuple(torch.randint(-3, 4, (2,)).tolist())
        pd[i] = torch.roll(src, shifts=shift, dims=(0, 1))
    return pd, gt


def _sample_inputs(max_instances: int) -> _Input:
    preds, target = [], []
    for _ in range(NUM_BATCHES):
        batch_preds, batch_target = [], []
        for _ in range(BATCH_SIZE):
            pd, gt = _generate_image_masks(max_instances)
            batch_preds.append(pd)
            batch_target.append(gt)
        preds.append(batch_preds)
        target.append(batch_target)
    return _Input(preds=preds, target=target)


_inputs_sparse = _sample_inputs(max_instances=3)
_inputs_dense = _sample_inputs(max_instances=6)


# --------------------------------------------------------------------------------------
# Main parametrized correctness tests
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("preds", "target"),
    [
        (_inputs_sparse.preds, _inputs_sparse.target),
        (_inputs_dense.preds, _inputs_dense.target),
    ],
)
@pytest.mark.parametrize("match_method", ["hungarian", "greedy"])
class TestSortedAP(MetricTester):
    """Test class for `SortedAP` metric."""

    atol = 1e-4

    @pytest.mark.parametrize("ddp", [True, False])
    def test_sorted_ap_class(self, preds, target, match_method, ddp):
        """Test module interface."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=SortedAP,
            reference_metric=partial(_reference_sorted_ap, match_method=match_method),
            metric_args={"match_method": match_method},
        )

    def test_sorted_ap_functional(self, preds, target, match_method):
        """Test functional interface."""
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=sorted_ap,
            reference_metric=partial(_reference_sorted_ap, match_method=match_method),
            metric_args={"match_method": match_method},
        )


# --------------------------------------------------------------------------------------
# Metric attributes
# --------------------------------------------------------------------------------------


def test_metric_attributes():
    """Sanity check the metric's static attributes."""
    assert SortedAP.is_differentiable is False
    assert SortedAP.higher_is_better is True
    assert SortedAP.full_state_update is False


# --------------------------------------------------------------------------------------
# Hand-crafted edge cases
# --------------------------------------------------------------------------------------


def _mask_at(height, width, y0, y1, x0, x1) -> Tensor:
    m = torch.zeros(1, height, width, dtype=torch.bool)
    m[0, y0:y1, x0:x1] = True
    return m


class TestSortedAPEdgeCases:
    """Edge case and input-validation tests for `sorted_ap` / `SortedAP`."""

    def test_perfect_match(self):
        """Identical preds and target should score 1."""
        mask = _mask_at(4, 4, 0, 2, 0, 2)
        assert sorted_ap([mask], [mask]) == torch.tensor(1.0)

    def test_no_overlap(self):
        """Fully disjoint preds and target should score 0."""
        preds = _mask_at(4, 4, 0, 2, 0, 2)
        target = _mask_at(4, 4, 2, 4, 2, 4)
        assert sorted_ap([preds], [target]) == torch.tensor(0.0)

    def test_empty_preds_and_target(self):
        """No predictions and no targets is a trivially perfect score."""
        empty = torch.zeros(0, 4, 4, dtype=torch.bool)
        assert sorted_ap([empty], [empty]) == torch.tensor(1.0)

    def test_predictions_without_targets(self):
        """Predictions with nothing to match against should score 0."""
        preds = torch.ones(1, 4, 4, dtype=torch.bool)
        empty = torch.zeros(0, 4, 4, dtype=torch.bool)
        assert sorted_ap([preds], [empty]) == torch.tensor(0.0)

    def test_targets_without_predictions(self):
        """Targets with no predictions to match against should score 0."""
        empty = torch.zeros(0, 4, 4, dtype=torch.bool)
        target = torch.ones(1, 4, 4, dtype=torch.bool)
        assert sorted_ap([empty], [target]) == torch.tensor(0.0)

    def test_empty_instance_channels_are_dropped(self):
        """All-zero mask channels should be ignored rather than counted as instances."""
        mask = _mask_at(4, 4, 0, 2, 0, 2)
        padded_preds = torch.cat([mask, torch.zeros(1, 4, 4, dtype=torch.bool)], dim=0)
        assert sorted_ap([padded_preds], [mask]) == torch.tensor(1.0)

    def test_invalid_match_method(self):
        """An unsupported `match_method` should raise a ValueError."""
        mask = _mask_at(4, 4, 0, 2, 0, 2)
        with pytest.raises(ValueError, match="Expected argument `match_method`.*"):
            sorted_ap([mask], [mask], match_method="invalid")

    def test_mismatched_number_of_images(self):
        """Different numbers of images in preds vs.

        target should raise a ValueError.

        """
        mask = _mask_at(4, 4, 0, 2, 0, 2)
        with pytest.raises(ValueError, match="Expected `preds` and `target` to have the same length.*"):
            sorted_ap([mask, mask], [mask])

    def test_wrong_number_of_dimensions(self):
        """Masks that are not 3D (num_instances, height, width) tensors should raise a ValueError."""
        with pytest.raises(ValueError, match="Expected each image in .*3D tensor.*"):
            sorted_ap([torch.zeros(4, 4, dtype=torch.bool)], [torch.zeros(1, 4, 4, dtype=torch.bool)])

    def test_mismatched_spatial_dimensions(self):
        """Preds/target with different (height, width) should raise a ValueError."""
        with pytest.raises(ValueError, match="Expected the spatial dimensions.*"):
            sorted_ap([torch.zeros(1, 4, 4, dtype=torch.bool)], [torch.zeros(1, 8, 8, dtype=torch.bool)])

    import importlib

    def test_hungarian_falls_back_to_greedy_without_scipy(self, monkeypatch):
        """Requesting hungarian matching without scipy installed should warn and fall back to greedy."""
        sorted_ap_module = importlib.import_module("torchmetrics.functional.segmentation.sorted_ap")
        monkeypatch.setattr(sorted_ap_module, "_SCIPY_AVAILABLE", False)
        mask = _mask_at(4, 4, 0, 2, 0, 2)
        with pytest.warns(UserWarning, match="Argument `match_method='hungarian'`.*"):
            sorted_ap([mask], [mask], match_method="hungarian")
