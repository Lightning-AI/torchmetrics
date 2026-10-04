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
from statsmodels.stats.inter_rater import fleiss_kappa as sk_fleiss_kappa

from torchmetrics.functional.nominal.fleiss_kappa import fleiss_kappa
from torchmetrics.nominal.fleiss_kappa import FleissKappa
from unittests import BATCH_SIZE, NUM_BATCHES, NUM_CLASSES
from unittests._helpers.testers import MetricTester

NUM_RATERS = 20
NUM_CATEGORIES = NUM_CLASSES


def _reference_fleiss_kappa(preds, target, mode):
    if mode == "probs":
        counts = np.zeros((preds.shape[0], preds.shape[1]))
        preds = preds.argmax(dim=1)
        for participant in range(preds.shape[0]):
            for rater in range(preds.shape[1]):
                counts[participant, preds[participant, rater]] += 1
        return sk_fleiss_kappa(counts)
    return sk_fleiss_kappa(preds)


def wrapped_fleiss_kappa(preds, target, mode):
    """Wrapped function for `fleiss_kappa` to support testing framework."""
    return fleiss_kappa(preds, mode)


class WrappedFleissKappa(FleissKappa):
    """Wrapped class for `FleissKappa` to support testing framework."""

    def update(self, preds, target):
        """Update function."""
        super().update(preds)


@pytest.mark.parametrize("num_raters", [2, 5, 20])
@pytest.mark.parametrize("majority_samples", [50, 99])
def test_fleiss_kappa_perfect_agreement(num_raters, majority_samples):
    """Perfect agreement must score exactly one even when chance agreement is high."""
    counts = torch.zeros(100, 2, dtype=torch.long)
    counts[:majority_samples, 0] = num_raters
    counts[majority_samples:, 1] = num_raters
    expected = torch.tensor(1.0)
    torch.testing.assert_close(fleiss_kappa(counts), expected, atol=0, rtol=0)
    metric = FleissKappa()
    for batch in counts.split(30):
        metric.update(batch)
    torch.testing.assert_close(metric.compute(), expected, atol=0, rtol=0)


def test_fleiss_kappa_complete_disagreement():
    """Complete disagreement must retain its exact negative score."""
    counts = torch.ones(20, 2, dtype=torch.long)
    torch.testing.assert_close(fleiss_kappa(counts), torch.tensor(-1.0), atol=0, rtol=0)


def test_fleiss_kappa_zero_denominator():
    """Removing bias must preserve the existing zero-denominator result."""
    counts = torch.full((10, 1), 3, dtype=torch.long)
    torch.testing.assert_close(fleiss_kappa(counts), torch.tensor(0.0))


def _random_counts(high, size):
    """Generate random counts matrix that is fully ranked.

    Interface is similar to torch.randint.

    """
    x = torch.randint(high=high, size=size)
    x_sum = x.sum(-1)
    x_total = x_sum.max()
    x[:, :, -1] = x_total - (x_sum - x[:, :, -1])
    return x


@pytest.mark.parametrize(
    ("preds", "target", "mode"),
    [  # target is not used in any of the functions
        (
            _random_counts(high=NUM_RATERS, size=(NUM_BATCHES, BATCH_SIZE, NUM_CATEGORIES)),
            _random_counts(high=NUM_RATERS, size=(NUM_BATCHES, BATCH_SIZE, NUM_CATEGORIES)),
            "counts",
        ),
        (
            torch.randn(NUM_BATCHES, BATCH_SIZE, NUM_CATEGORIES, NUM_RATERS),
            torch.randn(NUM_BATCHES, BATCH_SIZE, NUM_CATEGORIES, NUM_RATERS),
            "probs",
        ),
    ],
)
class TestFleissKappa(MetricTester):
    """Test class for `FleissKappa` metric."""

    atol = 1e-5

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_fleiss_kappa(self, ddp, preds, target, mode):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=WrappedFleissKappa,
            reference_metric=partial(_reference_fleiss_kappa, mode=mode),
            metric_args={"mode": mode},
        )

    def test_fleiss_kappa_functional(self, preds, target, mode):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=wrapped_fleiss_kappa,
            reference_metric=partial(_reference_fleiss_kappa, mode=mode),
            metric_args={"mode": mode},
        )

    def test_fleiss_kappa_differentiability(self, preds, target, mode):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        self.run_differentiability_test(
            preds,
            target,
            metric_module=WrappedFleissKappa,
            metric_functional=wrapped_fleiss_kappa,
            metric_args={"mode": mode},
        )
