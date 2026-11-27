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
    ("preds", "target"),
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


@pytest.mark.parametrize("ensemble_size", [0, 1])
def test_error_on_insufficient_ensemble_members(ensemble_size):
    """Empty ensembles must raise the same informative error as single-member ensembles."""
    with pytest.raises(ValueError, match="CRPS requires at least 2 ensemble members"):
        continuous_ranked_probability_score(torch.empty(3, ensemble_size), torch.ones(3))


@pytest.mark.parametrize("shape", [(), (3,), (3, 2, 2)])
def test_error_on_invalid_prediction_dimensions(shape):
    """Invalid ensemble dimensions must raise before indexing the ensemble axis."""
    with pytest.raises(ValueError, match="2D"):
        continuous_ranked_probability_score(torch.zeros(shape), torch.zeros(3))


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int8, torch.float32, torch.float64])
def test_crps_integer_and_tied_ensembles(dtype):
    """Sorted and tied ensemble members must agree with an independent pairwise definition."""
    preds = torch.tensor([[0, 100, 100, 0], [1, 1, 1, 1]], dtype=dtype)
    target = torch.tensor([50, 2], dtype=dtype)
    reference_preds, reference_target = preds.double(), target.double()
    expected = (
        (reference_preds - reference_target.unsqueeze(1)).abs().mean(dim=1)
        - (reference_preds.unsqueeze(2) - reference_preds.unsqueeze(1)).abs().mean(dim=(1, 2)) / 2
    ).mean()
    result = continuous_ranked_probability_score(preds, target)
    torch.testing.assert_close(result.double(), expected)
    metric = ContinuousRankedProbabilityScore().to(torch.float64)
    for pred_batch, target_batch in zip(preds.split(1), target.split(1)):
        metric.update(pred_batch, target_batch)
    torch.testing.assert_close(metric.compute(), expected)


def test_crps_large_ensemble():
    """A large uniformly spaced ensemble has a closed-form score without requiring pairwise distances."""
    ensemble_size = 4096
    preds = torch.arange(ensemble_size, dtype=torch.float64).unsqueeze(0)
    expected = torch.tensor((ensemble_size - 1) * (2 * ensemble_size - 1) / (6 * ensemble_size), dtype=torch.float64)
    torch.testing.assert_close(
        continuous_ranked_probability_score(preds, torch.zeros(1, dtype=torch.float64)), expected
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_crps_low_precision_large_values(dtype):
    """Ensemble differences must not overflow before averaging a representable score."""
    preds = torch.tensor([[-60000.0, 60000.0]], dtype=dtype)
    target = torch.zeros(1, dtype=dtype)
    expected = preds.double().abs().mean() / 2
    torch.testing.assert_close(continuous_ranked_probability_score(preds, target).double(), expected)


@pytest.mark.parametrize(("pred_dtype", "target_dtype"), [(torch.int64, torch.float64), (torch.float64, torch.int64)])
def test_crps_mixed_float64_integer_inputs(pred_dtype, target_dtype):
    """Mixed integer and double precision data must not be rounded to float32."""
    preds = torch.full((1, 2), 2**24 + 1, dtype=pred_dtype)
    target = torch.full((1,), 2**24 + 1, dtype=target_dtype)
    torch.testing.assert_close(
        continuous_ranked_probability_score(preds, target), torch.tensor(0.0, dtype=torch.float64)
    )
