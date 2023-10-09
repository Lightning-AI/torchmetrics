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
from typing import Callable, Tuple

import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torchmetrics.audio import PermutationInvariantTraining
from torchmetrics.functional.audio import (
    permutation_invariant_training,
    scale_invariant_signal_distortion_ratio,
    signal_noise_ratio,
)
from torchmetrics.functional.audio.pit import (
    _find_best_perm_by_exhaustive_method,
    _find_best_perm_by_linear_sum_assignment,
)

from unittests import BATCH_SIZE, NUM_BATCHES, _Input
from unittests.helpers import seed_all
from unittests.helpers.testers import MetricTester

seed_all(42)

TIME = 10


# three speaker examples to test _find_best_perm_by_linear_sum_assignment
inputs1 = _Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, 3, TIME),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE, 3, TIME),
)
# two speaker examples to test _find_best_perm_by_exhuastive_method
inputs2 = _Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, 2, TIME),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE, 2, TIME),
)


def naive_implementation_pit_scipy(
    preds: Tensor,
    target: Tensor,
    metric_func: Callable,
    eval_func: str,
) -> Tuple[Tensor, Tensor]:
    """Naive implementation of `Permutation Invariant Training` based on Scipy.

    Args:
        preds: predictions, shape[batch, spk, time]
        target: targets, shape[batch, spk, time]
        metric_func: which metric
        eval_func: min or max

    Returns:
        best_metric:
            shape [batch]
        best_perm:
            shape [batch, spk]

    """
    batch_size, spk_num = target.shape[0:2]
    metric_mtx = torch.empty((batch_size, spk_num, spk_num), device=target.device)
    for t in range(spk_num):
        for e in range(spk_num):
            metric_mtx[:, t, e] = metric_func(preds[:, e, ...], target[:, t, ...])

    metric_mtx = metric_mtx.detach().cpu().numpy()
    best_metrics = []
    best_perms = []
    for b in range(batch_size):
        row_idx, col_idx = linear_sum_assignment(metric_mtx[b, ...], eval_func == "max")
        best_metrics.append(metric_mtx[b, row_idx, col_idx].mean())
        best_perms.append(col_idx)
    return torch.from_numpy(np.stack(best_metrics)), torch.from_numpy(np.stack(best_perms))


def _average_metric(preds: Tensor, target: Tensor, metric_func: Callable) -> Tensor:
    """Average the metric values.

    Args:
        preds: predictions, shape[batch, spk, time]
        target: targets, shape[batch, spk, time]
        metric_func: a function which return best_metric and best_perm

    Returns:
        the average of best_metric

    """
    return metric_func(preds, target)[0].mean()


snr_pit_scipy = partial(naive_implementation_pit_scipy, metric_func=signal_noise_ratio, eval_func="max")
si_sdr_pit_scipy = partial(
    naive_implementation_pit_scipy, metric_func=scale_invariant_signal_distortion_ratio, eval_func="max"
)


@pytest.mark.parametrize(
    "preds, target, ref_metric, metric_func, mode, eval_func",
    [
        (inputs1.preds, inputs1.target, snr_pit_scipy, signal_noise_ratio, "speaker-wise", "max"),
        (
            inputs1.preds,
            inputs1.target,
            si_sdr_pit_scipy,
            scale_invariant_signal_distortion_ratio,
            "speaker-wise",
            "max",
        ),
        (inputs2.preds, inputs2.target, snr_pit_scipy, signal_noise_ratio, "speaker-wise", "max"),
        (
            inputs2.preds,
            inputs2.target,
            si_sdr_pit_scipy,
            scale_invariant_signal_distortion_ratio,
            "speaker-wise",
            "max",
        ),
        (inputs1.preds, inputs1.target, snr_pit_scipy, signal_noise_ratio, "permutation-wise", "max"),
        (
            inputs1.preds,
            inputs1.target,
            si_sdr_pit_scipy,
            scale_invariant_signal_distortion_ratio,
            "permutation-wise",
            "max",
        ),
        (inputs2.preds, inputs2.target, snr_pit_scipy, signal_noise_ratio, "permutation-wise", "max"),
        (
            inputs2.preds,
            inputs2.target,
            si_sdr_pit_scipy,
            scale_invariant_signal_distortion_ratio,
            "permutation-wise",
            "max",
        ),
    ],
)
class TestPIT(MetricTester):
    """Test class for `PermutationInvariantTraining` metric."""

    atol = 1e-2

    @pytest.mark.parametrize("ddp", [True, False])
    def test_pit(self, preds, target, ref_metric, metric_func, mode, eval_func, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            PermutationInvariantTraining,
            reference_metric=partial(_average_metric, metric_func=ref_metric),
            metric_args={"metric_func": metric_func, "mode": mode, "eval_func": eval_func},
        )

    def test_pit_functional(self, preds, target, ref_metric, metric_func, mode, eval_func):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=permutation_invariant_training,
            reference_metric=ref_metric,
            metric_args={"metric_func": metric_func, "mode": mode, "eval_func": eval_func},
        )

    def test_pit_differentiability(self, preds, target, ref_metric, metric_func, mode, eval_func):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""

        def pit_diff(preds, target, metric_func, mode, eval_func):
            return permutation_invariant_training(preds, target, metric_func, mode, eval_func)[0]

        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=PermutationInvariantTraining,
            metric_functional=pit_diff,
            metric_args={"metric_func": metric_func, "mode": mode, "eval_func": eval_func},
        )

    def test_pit_half_cpu(self, preds, target, ref_metric, metric_func, mode, eval_func):
        """Test dtype support of the metric on CPU."""
        pytest.xfail("PIT metric does not support cpu + half precision")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_pit_half_gpu(self, preds, target, ref_metric, metric_func, mode, eval_func):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=PermutationInvariantTraining,
            metric_functional=partial(permutation_invariant_training, metric_func=metric_func, eval_func=eval_func),
            metric_args={"metric_func": metric_func, "mode": mode, "eval_func": eval_func},
        )


def test_error_on_different_shape() -> None:
    """Test that error is raised on different shapes of input."""
    metric = PermutationInvariantTraining(signal_noise_ratio)
    with pytest.raises(
        RuntimeError,
        match="Predictions and targets are expected to have the same shape at the batch and speaker dimensions",
    ):
        metric(torch.randn(3, 3, 10), torch.randn(3, 2, 10))


def test_error_on_wrong_eval_func() -> None:
    """Test that error is raised on wrong `eval_func` argument."""
    metric = PermutationInvariantTraining(signal_noise_ratio, eval_func="xxx")
    with pytest.raises(ValueError, match='eval_func can only be "max" or "min"'):
        metric(torch.randn(3, 3, 10), torch.randn(3, 3, 10))


def test_error_on_wrong_mode() -> None:
    """Test that error is raised on wrong `mode` argument."""
    metric = PermutationInvariantTraining(signal_noise_ratio, mode="xxx")
    with pytest.raises(ValueError, match='mode can only be "speaker-wise" or "permutation-wise"*'):
        metric(torch.randn(3, 3, 10), torch.randn(3, 3, 10))


def test_error_on_wrong_shape() -> None:
    """Test that error is raised on wrong input shape."""
    metric = PermutationInvariantTraining(signal_noise_ratio)
    with pytest.raises(ValueError, match="Inputs must be of shape *"):
        metric(torch.randn(3), torch.randn(3))


def test_consistency_of_two_implementations() -> None:
    """Test that both backend functions for computing metric (depending on torch version) returns the same result."""
    shapes_test = [(5, 2, 2), (4, 3, 3), (4, 4, 4), (3, 5, 5)]
    for shp in shapes_test:
        metric_mtx = torch.randn(size=shp)
        bm1, bp1 = _find_best_perm_by_linear_sum_assignment(metric_mtx, torch.max)
        bm2, bp2 = _find_best_perm_by_exhaustive_method(metric_mtx, torch.max)
        assert torch.allclose(bm1, bm2)
        assert (bp1 == bp2).all()
