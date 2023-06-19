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
from statsmodels.stats.inter_rater import aggregate_raters
from statsmodels.stats.inter_rater import fleiss_kappa as sk_fleiss_kappa
from torchmetrics.functional.nominal.fleiss_kappa import fleiss_kappa
from torchmetrics.nominal.fleiss_kappa import FleissKappa

from unittests import BATCH_SIZE, NUM_BATCHES, NUM_CLASSES
from unittests.helpers.testers import MetricTester

NUM_RATERS = 20
NUM_CATEGORIES = NUM_CLASSES


def _compare_func(preds, target, mode):
    if mode == "counts":
        preds = preds.argmax(dim=1)
        target = target.argmax(dim=1)
    return sk_fleiss_kappa(aggregate_raters(preds))


def wrapped_fleiss_kappa(preds, target, mode):
    """Wrapped function for `fleiss_kappa` to support testing framework."""
    return fleiss_kappa(preds, mode)


class WrappedFleissKappa(FleissKappa):
    """Wrapped class for `FleissKappa` to support testing framework."""

    def update(self, preds, target):
        """Update function."""
        super().update(preds)


@pytest.mark.parametrize(
    "preds, target, mode",
    [  # target is not used in any of the functions
        (
            torch.randint(high=NUM_RATERS, size=(NUM_BATCHES, BATCH_SIZE, NUM_CATEGORIES)),
            torch.randint(high=NUM_RATERS, size=(NUM_BATCHES, BATCH_SIZE, NUM_CATEGORIES)),
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

    @pytest.mark.parametrize("ddp", [False, True])
    def test_fleiss_kappa(self, ddp, preds, target, mode):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=WrappedFleissKappa,
            reference_metric=partial(_compare_func, mode=mode),
            metric_args={"mode": mode},
        )

    def test_fleiss_kappa_functional(self, preds, target, mode):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=wrapped_fleiss_kappa,
            reference_metric=partial(_compare_func, mode=mode),
            metric_args={"mode": mode},
        )

    def test_fleiss_kappa_differentiability(self, preds, target, mode):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        metric_args = {"num_classes": NUM_CLASSES}
        self.run_differentiability_test(
            preds,
            target,
            metric_module=WrappedFleissKappa,
            metric_functional=wrapped_fleiss_kappa,
            metric_args=metric_args,
        )
