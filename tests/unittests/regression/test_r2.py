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
from sklearn.metrics import r2_score as sk_r2score
from torchmetrics.functional import r2_score
from torchmetrics.regression import R2Score

from unittests import BATCH_SIZE, NUM_BATCHES, _Input
from unittests.helpers import seed_all
from unittests.helpers.testers import MetricTester

seed_all(42)

num_targets = 5


_single_target_inputs = _Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE),
)

_multi_target_inputs = _Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, num_targets),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE, num_targets),
)


def _single_target_ref_metric(preds, target, adjusted, multioutput):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()
    r2_score = sk_r2score(sk_target, sk_preds, multioutput=multioutput)
    if adjusted != 0:
        return 1 - (1 - r2_score) * (sk_preds.shape[0] - 1) / (sk_preds.shape[0] - adjusted - 1)
    return r2_score


def _multi_target_ref_metric(preds, target, adjusted, multioutput):
    sk_preds = preds.view(-1, num_targets).numpy()
    sk_target = target.view(-1, num_targets).numpy()
    r2_score = sk_r2score(sk_target, sk_preds, multioutput=multioutput)
    if adjusted != 0:
        return 1 - (1 - r2_score) * (sk_preds.shape[0] - 1) / (sk_preds.shape[0] - adjusted - 1)
    return r2_score


@pytest.mark.parametrize("adjusted", [0, 5, 10])
@pytest.mark.parametrize("multioutput", ["raw_values", "uniform_average", "variance_weighted"])
@pytest.mark.parametrize(
    "preds, target, ref_metric, num_outputs",
    [
        (_single_target_inputs.preds, _single_target_inputs.target, _single_target_ref_metric, 1),
        (_multi_target_inputs.preds, _multi_target_inputs.target, _multi_target_ref_metric, num_targets),
    ],
)
class TestR2Score(MetricTester):
    """Test class for `R2Score` metric."""

    @pytest.mark.parametrize("ddp", [True, False])
    def test_r2(self, adjusted, multioutput, preds, target, ref_metric, num_outputs, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            R2Score,
            partial(ref_metric, adjusted=adjusted, multioutput=multioutput),
            metric_args={"adjusted": adjusted, "multioutput": multioutput, "num_outputs": num_outputs},
        )

    def test_r2_functional(self, adjusted, multioutput, preds, target, ref_metric, num_outputs):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            r2_score,
            partial(ref_metric, adjusted=adjusted, multioutput=multioutput),
            metric_args={"adjusted": adjusted, "multioutput": multioutput},
        )

    def test_r2_differentiability(self, adjusted, multioutput, preds, target, ref_metric, num_outputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=partial(R2Score, num_outputs=num_outputs),
            metric_functional=r2_score,
            metric_args={"adjusted": adjusted, "multioutput": multioutput},
        )

    def test_r2_half_cpu(self, adjusted, multioutput, preds, target, ref_metric, num_outputs):
        """Test dtype support of the metric on CPU."""
        self.run_precision_test_cpu(
            preds,
            target,
            partial(R2Score, num_outputs=num_outputs),
            r2_score,
            {"adjusted": adjusted, "multioutput": multioutput},
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_r2_half_gpu(self, adjusted, multioutput, preds, target, ref_metric, num_outputs):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(
            preds,
            target,
            partial(R2Score, num_outputs=num_outputs),
            r2_score,
            {"adjusted": adjusted, "multioutput": multioutput},
        )


def test_error_on_different_shape(metric_class=R2Score):
    """Test that error is raised on different shapes of input."""
    metric = metric_class()
    with pytest.raises(RuntimeError, match="Predictions and targets are expected to have the same shape"):
        metric(torch.randn(100), torch.randn(50))


def test_error_on_multidim_tensors(metric_class=R2Score):
    """Test that error is raised if a larger than 2D tensor is given as input."""
    metric = metric_class()
    with pytest.raises(
        ValueError,
        match=r"Expected both prediction and target to be 1D or 2D tensors, but received tensors with dimension .",
    ):
        metric(torch.randn(10, 20, 5), torch.randn(10, 20, 5))


def test_error_on_too_few_samples(metric_class=R2Score):
    """Test that error is raised if too few samples are provided."""
    metric = metric_class()
    with pytest.raises(ValueError, match="Needs at least two samples to calculate r2 score."):
        metric(torch.randn(1), torch.randn(1))
    metric.reset()

    # calling update twice should still work
    metric.update(torch.randn(1), torch.randn(1))
    metric.update(torch.randn(1), torch.randn(1))
    assert metric.compute()


def test_warning_on_too_large_adjusted(metric_class=R2Score):
    """Test that warning is raised if adjusted argument is set to more than or equal to the number of datapoints."""
    metric = metric_class(adjusted=10)

    with pytest.warns(
        UserWarning,
        match="More independent regressions than data points in adjusted r2 score. Falls back to standard r2 score.",
    ):
        metric(torch.randn(10), torch.randn(10))

    with pytest.warns(UserWarning, match="Division by zero in adjusted r2 score. Falls back to standard r2 score."):
        metric(torch.randn(11), torch.randn(11))


def test_constant_target():
    """Check for a near constant target that a value of 0 is returned."""
    y_true = torch.tensor([-5.1608, -5.1609, -5.1608, -5.1608, -5.1608, -5.1608])
    y_pred = torch.tensor([-3.9865, -5.4648, -5.0238, -4.3899, -5.6672, -4.7336])
    score = r2_score(preds=y_pred, target=y_true)
    assert score == 0
