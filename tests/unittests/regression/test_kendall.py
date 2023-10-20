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
import operator
from functools import partial

import pytest
import torch
from lightning_utilities.core.imports import compare_version
from scipy.stats import kendalltau
from torchmetrics.functional.regression.kendall import kendall_rank_corrcoef
from torchmetrics.regression.kendall import KendallRankCorrCoef

from unittests import BATCH_SIZE, EXTRA_DIM, NUM_BATCHES, _Input
from unittests.helpers import seed_all
from unittests.helpers.testers import MetricTester

_SCIPY_GREATER_EQUAL_1_8 = compare_version("scipy", operator.ge, "1.8.0")

seed_all(42)


_single_inputs1 = _Input(preds=torch.rand(NUM_BATCHES, BATCH_SIZE), target=torch.rand(NUM_BATCHES, BATCH_SIZE))
_single_inputs2 = _Input(preds=torch.randn(NUM_BATCHES, BATCH_SIZE), target=torch.randn(NUM_BATCHES, BATCH_SIZE))
_single_inputs3 = _Input(
    preds=torch.randint(-10, 10, (NUM_BATCHES, BATCH_SIZE)), target=torch.randint(-10, 10, (NUM_BATCHES, BATCH_SIZE))
)
_multi_inputs1 = _Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM), target=torch.rand(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)
)
_multi_inputs2 = _Input(
    preds=torch.randn(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM), target=torch.randn(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)
)
_multi_inputs3 = _Input(
    preds=torch.randint(-10, 10, (NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)),
    target=torch.randint(-10, 10, (NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)),
)


def _scipy_kendall(preds, target, alternative, variant):
    metric_args = {}
    if _SCIPY_GREATER_EQUAL_1_8:
        metric_args = {"alternative": alternative or "two-sided"}  # scipy cannot accept `None`
    if preds.ndim == 2:
        out = [
            kendalltau(p.numpy(), t.numpy(), method="asymptotic", variant=variant, **metric_args)
            for p, t in zip(preds.T, target.T)
        ]
        tau = torch.cat([torch.tensor(o[0]).unsqueeze(0) for o in out])
        p_value = torch.cat([torch.tensor(o[1]).unsqueeze(0) for o in out])
        if alternative is not None:
            return tau, p_value
        return tau

    tau, p_value = kendalltau(preds.numpy(), target.numpy(), method="asymptotic", variant=variant, **metric_args)

    if alternative is not None:
        return torch.tensor(tau), torch.tensor(p_value)
    return torch.tensor(tau)


@pytest.mark.parametrize(
    "preds, target, alternative",
    [
        (_single_inputs1.preds, _single_inputs1.target, None),
        (_single_inputs2.preds, _single_inputs2.target, "less"),
        (_single_inputs3.preds, _single_inputs3.target, "greater"),
        (_multi_inputs1.preds, _multi_inputs1.target, None),
        (_multi_inputs2.preds, _multi_inputs2.target, "two-sided"),
        (_multi_inputs3.preds, _multi_inputs3.target, "greater"),
    ],
)
@pytest.mark.parametrize("variant", ["b", "c"])
class TestKendallRankCorrCoef(MetricTester):
    """Test class for `KendallRankCorrCoef` metric."""

    @pytest.mark.parametrize("ddp", [False, True])
    def test_kendall_rank_corrcoef(self, preds, target, alternative, variant, ddp):
        """Test class implementation of metric."""
        num_outputs = EXTRA_DIM if preds.ndim == 3 else 1
        t_test = bool(alternative is not None)
        _sk_kendall_tau = partial(_scipy_kendall, alternative=alternative, variant=variant)
        alternative = _adjust_alternative_to_scipy(alternative)

        self.run_class_metric_test(
            ddp,
            preds,
            target,
            KendallRankCorrCoef,
            _sk_kendall_tau,
            metric_args={"t_test": t_test, "alternative": alternative, "variant": variant, "num_outputs": num_outputs},
        )

    def test_kendall_rank_corrcoef_functional(self, preds, target, alternative, variant):
        """Test functional implementation of metric."""
        t_test = bool(alternative is not None)
        alternative = _adjust_alternative_to_scipy(alternative)
        metric_args = {"t_test": t_test, "alternative": alternative, "variant": variant}
        _sk_kendall_tau = partial(_scipy_kendall, alternative=alternative, variant=variant)
        self.run_functional_metric_test(preds, target, kendall_rank_corrcoef, _sk_kendall_tau, metric_args=metric_args)

    def test_kendall_rank_corrcoef_differentiability(self, preds, target, alternative, variant):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        num_outputs = EXTRA_DIM if preds.ndim == 3 else 1
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=partial(KendallRankCorrCoef, num_outputs=num_outputs),
            metric_functional=kendall_rank_corrcoef,
        )


def _adjust_alternative_to_scipy(alternative):
    """Scipy<1.8.0 supports only two-sided hypothesis testing."""
    if alternative is not None and not compare_version("scipy", operator.ge, "1.8.0"):
        return "two-sided"
    return alternative
