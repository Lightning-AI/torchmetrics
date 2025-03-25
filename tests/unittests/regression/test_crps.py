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
import pytest
import torch
from properscoring import crps_ensemble

from torchmetrics.functional.regression.crps import continuous_ranked_probability_score
from torchmetrics.regression.crps import ContinuousRankedProbabilityScore
from unittests import BATCH_SIZE, NUM_BATCHES, _Input
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

seed_all(42)

_input_10ensemble = _Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, 10),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE),
)

_input2_5ensemble = _Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, 5),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE),
)


def _reference_implementation(preds, target):
    sk_preds = preds.numpy()
    sk_target = target.numpy()
    return crps_ensemble(sk_target, sk_preds).mean()


@pytest.mark.parametrize(
    "preds, target",
    [
        (_input2_5ensemble.preds, _input2_5ensemble.target),
        (_input_10ensemble.preds, _input_10ensemble.target),
    ],
)
class TestContinuousRankedProbabilityScore(MetricTester):
    """Test class for `ContinuousRankedProbabilityScore` metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_continuous_ranked_probability_score(self, preds, target, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=ContinuousRankedProbabilityScore,
            reference_metric=_reference_implementation,
        )

    def test_continuous_ranked_probability_score_functional(self, preds, target):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=continuous_ranked_probability_score,
            reference_metric=_reference_implementation,
        )


def test_error_on_different_shape(metric_class=ContinuousRankedProbabilityScore):
    """Test that error is raised on different shapes of input."""
    metric = metric_class()
    with pytest.raises(RuntimeError, match="Predictions and targets are expected to have the same shape"):
        metric(torch.randn(100, 5), torch.randn(50))


def test_error_on_single_ensemble_member():
    """Test that error is raised on single ensemble member."""
    metric = ContinuousRankedProbabilityScore()
    with pytest.raises(ValueError, match="CRPS requires at least 2 ensemble members, but.*"):
        metric(torch.randn(100, 1), torch.randn(100))
