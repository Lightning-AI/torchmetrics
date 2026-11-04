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
import sys
from functools import partial
from itertools import combinations

import pytest
import torch
from lightning_utilities.core.imports import compare_version
from scipy.stats import kendalltau

from torchmetrics.functional.regression.kendall import kendall_rank_corrcoef
from torchmetrics.regression.kendall import KendallRankCorrCoef
from torchmetrics.utilities.imports import _SCIPY_GREATER_EQUAL_1_8
from unittests import BATCH_SIZE, EXTRA_DIM, NUM_BATCHES, _Input
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

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
# few distinct values, so most pairs are tied in at least one of the sequences
_tied_single_inputs = _Input(
    preds=torch.randint(0, 4, (NUM_BATCHES, BATCH_SIZE)).float(),
    target=torch.randint(0, 4, (NUM_BATCHES, BATCH_SIZE)).float(),
)
_tied_multi_inputs = _Input(
    preds=torch.randint(0, 4, (NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)).float(),
    target=torch.randint(0, 4, (NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)).float(),
)


def _reference_scipy_kendall(preds, target, alternative, variant):
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
    ("preds", "target", "alternative"),
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

    @pytest.mark.skipif(sys.platform == "darwin", reason="Fails on MacOS")  # TODO: investigate
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_kendall_rank_corrcoef(self, preds, target, alternative, variant, ddp):
        """Test class implementation of metric."""
        num_outputs = EXTRA_DIM if preds.ndim == 3 else 1
        t_test = bool(alternative is not None)
        _sk_kendall_tau = partial(_reference_scipy_kendall, alternative=alternative, variant=variant)
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
        _sk_kendall_tau = partial(_reference_scipy_kendall, alternative=alternative, variant=variant)
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


def _reference_tau_a(preds, target):
    """Brute-force Kendall's tau-a: ``(C - D) / (n * (n - 1) / 2)``, counting every pair, tied ones included.

    Scipy only implements tau-b and tau-c, so tau-a is computed from its definition.

    """
    if preds.ndim == 2:
        return torch.stack([_reference_tau_a(p, t) for p, t in zip(preds.T, target.T)])
    preds, target = preds.tolist(), target.tolist()
    concordant = discordant = 0
    for i, j in combinations(range(len(preds)), 2):
        product = (preds[i] - preds[j]) * (target[i] - target[j])
        concordant += product > 0
        discordant += product < 0
    return torch.tensor((concordant - discordant) / (len(preds) * (len(preds) - 1) / 2))


@pytest.mark.parametrize(
    ("preds", "target"),
    [
        (_single_inputs1.preds, _single_inputs1.target),
        (_tied_single_inputs.preds, _tied_single_inputs.target),
        (_multi_inputs1.preds, _multi_inputs1.target),
        (_tied_multi_inputs.preds, _tied_multi_inputs.target),
    ],
)
class TestKendallRankCorrCoefTauA(MetricTester):
    """Test `KendallRankCorrCoef` with `variant="a"` against the definition of tau-a."""

    @pytest.mark.skipif(sys.platform == "darwin", reason="Fails on MacOS")  # TODO: investigate
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_kendall_rank_corrcoef(self, preds, target, ddp):
        """Test class implementation of metric."""
        num_outputs = EXTRA_DIM if preds.ndim == 3 else 1
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            KendallRankCorrCoef,
            _reference_tau_a,
            metric_args={"variant": "a", "num_outputs": num_outputs},
        )

    def test_kendall_rank_corrcoef_functional(self, preds, target):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds, target, kendall_rank_corrcoef, _reference_tau_a, metric_args={"variant": "a"}
        )


def test_kendall_tau_a_with_ties():
    """Tied pairs stay in the denominator of tau-a; dropping them would give Goodman-Kruskal's gamma instead."""
    preds = torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    target = torch.tensor([1.0, 2.0, 2.0, 3.0, 3.0, 4.0])
    # 10 concordant pairs, no discordant pair and 5 tied pairs out of 15
    assert kendall_rank_corrcoef(preds, target, variant="a").item() == pytest.approx(10 / 15)
    assert KendallRankCorrCoef(variant="a")(preds, target).item() == pytest.approx(10 / 15)


def test_kendall_tau_a_equals_tau_b_without_ties():
    """Without ties tau-a and tau-b coincide, so scipy's tau-b serves as a cross-check for tau-a."""
    preds, target = _single_inputs2.preds[0], _single_inputs2.target[0]
    expected = kendalltau(preds.numpy(), target.numpy(), variant="b")[0]
    assert kendall_rank_corrcoef(preds, target, variant="a").item() == pytest.approx(expected, abs=1e-6)


def _adjust_alternative_to_scipy(alternative):
    """Scipy<1.8.0 supports only two-sided hypothesis testing."""
    if alternative is not None and not compare_version("scipy", operator.ge, "1.8.0"):
        return "two-sided"
    return alternative
