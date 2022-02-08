# Copyright The PyTorch Lightning team.
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
from collections import namedtuple
from functools import partial
from typing import Callable, Tuple

import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from tests.helpers import seed_all
from tests.helpers.testers import BATCH_SIZE, NUM_BATCHES, MetricTester
from torchmetrics.audio import PermutationInvariantTraining
from torchmetrics.functional import (
    permutation_invariant_training,
    scale_invariant_signal_distortion_ratio,
    signal_noise_ratio,
)
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_1_6

seed_all(42)

TIME = 10

Input = namedtuple("Input", ["preds", "target"])

# three speaker examples to test _find_best_perm_by_linear_sum_assignment
inputs1 = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, 3, TIME),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE, 3, TIME),
)
# two speaker examples to test _find_best_perm_by_exhuastive_method
inputs2 = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, 2, TIME),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE, 2, TIME),
)


def naive_implementation_pit_scipy(
    preds: Tensor,
    target: Tensor,
    metric_func: Callable,
    eval_func: str,
) -> Tuple[Tensor, Tensor]:
    """A naive implementation of `Permutation Invariant Training` based on Scipy.

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

    # pit_r = PermutationInvariantTraining(metric_func, eval_func)(preds, target)
    metric_mtx = metric_mtx.detach().cpu().numpy()
    best_metrics = []
    best_perms = []
    for b in range(batch_size):
        row_idx, col_idx = linear_sum_assignment(metric_mtx[b, ...], eval_func == "max")
        best_metrics.append(metric_mtx[b, row_idx, col_idx].mean())
        best_perms.append(col_idx)
    return torch.from_numpy(np.stack(best_metrics)), torch.from_numpy(np.stack(best_perms))


def _average_metric(preds: Tensor, target: Tensor, metric_func: Callable) -> Tensor:
    """average the metric values.

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
    "preds, target, sk_metric, metric_func, eval_func",
    [
        (inputs1.preds, inputs1.target, snr_pit_scipy, signal_noise_ratio, "max"),
        (inputs1.preds, inputs1.target, si_sdr_pit_scipy, scale_invariant_signal_distortion_ratio, "max"),
        (inputs2.preds, inputs2.target, snr_pit_scipy, signal_noise_ratio, "max"),
        (inputs2.preds, inputs2.target, si_sdr_pit_scipy, scale_invariant_signal_distortion_ratio, "max"),
    ],
)
class TestPIT(MetricTester):
    atol = 1e-2

    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_pit(self, preds, target, sk_metric, metric_func, eval_func, ddp, dist_sync_on_step):
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            PermutationInvariantTraining,
            sk_metric=partial(_average_metric, metric_func=sk_metric),
            dist_sync_on_step=dist_sync_on_step,
            metric_args=dict(metric_func=metric_func, eval_func=eval_func),
        )

    def test_pit_functional(self, preds, target, sk_metric, metric_func, eval_func):
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=permutation_invariant_training,
            sk_metric=sk_metric,
            metric_args=dict(metric_func=metric_func, eval_func=eval_func),
        )

    def test_pit_differentiability(self, preds, target, sk_metric, metric_func, eval_func):
        def pit_diff(preds, target, metric_func, eval_func):
            return permutation_invariant_training(preds, target, metric_func, eval_func)[0]

        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=PermutationInvariantTraining,
            metric_functional=pit_diff,
            metric_args={"metric_func": metric_func, "eval_func": eval_func},
        )

    @pytest.mark.skipif(
        not _TORCH_GREATER_EQUAL_1_6, reason="half support of core operations on not support before pytorch v1.6"
    )
    def test_pit_half_cpu(self, preds, target, sk_metric, metric_func, eval_func):
        pytest.xfail("PIT metric does not support cpu + half precision")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_pit_half_gpu(self, preds, target, sk_metric, metric_func, eval_func):
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=PermutationInvariantTraining,
            metric_functional=partial(permutation_invariant_training, metric_func=metric_func, eval_func=eval_func),
            metric_args={"metric_func": metric_func, "eval_func": eval_func},
        )


def test_error_on_different_shape() -> None:
    metric = PermutationInvariantTraining(signal_noise_ratio, "max")
    with pytest.raises(RuntimeError, match="Predictions and targets are expected to have the same shape"):
        metric(torch.randn(3, 3, 10), torch.randn(3, 2, 10))


def test_error_on_wrong_eval_func() -> None:
    metric = PermutationInvariantTraining(signal_noise_ratio, "xxx")
    with pytest.raises(ValueError, match='eval_func can only be "max" or "min"'):
        metric(torch.randn(3, 3, 10), torch.randn(3, 3, 10))


def test_error_on_wrong_shape() -> None:
    metric = PermutationInvariantTraining(signal_noise_ratio, "max")
    with pytest.raises(ValueError, match="Inputs must be of shape *"):
        metric(torch.randn(3), torch.randn(3))


def test_consistency_of_two_implementations() -> None:
    from torchmetrics.functional.audio.pit import (
        _find_best_perm_by_exhuastive_method,
        _find_best_perm_by_linear_sum_assignment,
    )

    shapes_test = [(5, 2, 2), (4, 3, 3), (4, 4, 4), (3, 5, 5)]
    for shp in shapes_test:
        metric_mtx = torch.randn(size=shp)
        bm1, bp1 = _find_best_perm_by_linear_sum_assignment(metric_mtx, torch.max)
        bm2, bp2 = _find_best_perm_by_exhuastive_method(metric_mtx, torch.max)
        assert torch.allclose(bm1, bm2)
        assert (bp1 == bp2).all()
