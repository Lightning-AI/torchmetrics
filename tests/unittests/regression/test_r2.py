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
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

seed_all(42)

NUM_TARGETS = 5


_single_target_inputs = _Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE),
)

_multi_target_inputs = _Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, NUM_TARGETS),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE, NUM_TARGETS),
)


def _single_target_ref_wrapper(preds, target, adjusted, multioutput):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()
    r2_score = sk_r2score(sk_target, sk_preds, multioutput=multioutput)
    if adjusted != 0:
        return 1 - (1 - r2_score) * (sk_preds.shape[0] - 1) / (sk_preds.shape[0] - adjusted - 1)
    return r2_score


def _multi_target_ref_wrapper(preds, target, adjusted, multioutput):
    sk_preds = preds.view(-1, NUM_TARGETS).numpy()
    sk_target = target.view(-1, NUM_TARGETS).numpy()
    r2_score = sk_r2score(sk_target, sk_preds, multioutput=multioutput)
    if adjusted != 0:
        return 1 - (1 - r2_score) * (sk_preds.shape[0] - 1) / (sk_preds.shape[0] - adjusted - 1)
    return r2_score


@pytest.mark.parametrize("adjusted", [0, 5, 10])
@pytest.mark.parametrize("multioutput", ["raw_values", "uniform_average", "variance_weighted"])
@pytest.mark.parametrize(
    ("preds", "target", "ref_metric"),
    [
        (_single_target_inputs.preds, _single_target_inputs.target, _single_target_ref_wrapper),
        (_multi_target_inputs.preds, _multi_target_inputs.target, _multi_target_ref_wrapper),
    ],
)
class TestR2Score(MetricTester):
    """Test class for `R2Score` metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    @pytest.mark.parametrize("scale", [1.0, 1e-3])
    def test_r2(self, adjusted, multioutput, preds, target, ref_metric, ddp, scale):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds * scale,
            target * scale,
            R2Score,
            partial(ref_metric, adjusted=adjusted, multioutput=multioutput),
            metric_args={"adjusted": adjusted, "multioutput": multioutput},
        )

    def test_r2_functional(self, adjusted, multioutput, preds, target, ref_metric):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            r2_score,
            partial(ref_metric, adjusted=adjusted, multioutput=multioutput),
            metric_args={"adjusted": adjusted, "multioutput": multioutput},
        )

    def test_r2_differentiability(self, adjusted, multioutput, preds, target, ref_metric):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        self.run_differentiability_test(
            preds, target, R2Score, r2_score, {"adjusted": adjusted, "multioutput": multioutput}
        )

    def test_r2_half_cpu(self, adjusted, multioutput, preds, target, ref_metric):
        """Test dtype support of the metric on CPU."""
        self.run_precision_test_cpu(
            preds, target, R2Score, r2_score, {"adjusted": adjusted, "multioutput": multioutput}
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_r2_half_gpu(self, adjusted, multioutput, preds, target, ref_metric):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(
            preds, target, R2Score, r2_score, {"adjusted": adjusted, "multioutput": multioutput}
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


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_constant_target(scale):
    """Check for a near constant target that a value of 0 is returned."""
    y_true = torch.tensor([-5.1608, -5.1609, -5.1608, -5.1608, -5.1608, -5.1608]) * scale
    y_pred = torch.tensor([-3.9865, -5.4648, -5.0238, -4.3899, -5.6672, -4.7336]) * scale
    score = r2_score(preds=y_pred, target=y_true)
    assert score == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scale", [1.0, 1e-3, 1e-6])
@pytest.mark.parametrize("repeats", [1, 100])
@pytest.mark.parametrize("multioutput", ["raw_values", "uniform_average", "variance_weighted"])
def test_r2_scale_and_sample_count(dtype, scale, repeats, multioutput):
    """Rescaling or repeating samples must preserve mean, worse and perfect prediction scores."""
    target = torch.arange(4, dtype=dtype).unsqueeze(1) * torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
    preds = torch.stack([target[:, 0].mean().expand(4), torch.zeros(4, dtype=dtype), target[:, 2]], dim=1)
    expected = torch.as_tensor(sk_r2score(target.numpy(), preds.numpy(), multioutput=multioutput), dtype=dtype)
    target = (target * scale).repeat(repeats, 1)
    preds = (preds * scale).repeat(repeats, 1)

    score = r2_score(preds, target, multioutput=multioutput)
    torch.testing.assert_close(score, expected, atol=1e-4, rtol=1e-4)

    metric = R2Score(multioutput=multioutput).set_dtype(dtype)
    for pred_batch, target_batch in zip(preds.split(2), target.split(2)):
        metric.update(pred_batch, target_batch)
    torch.testing.assert_close(metric.compute(), expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("value", [0.0, 1e-3, 0.1, 1e3])
@pytest.mark.parametrize("num_samples", [4, 1000])
def test_r2_constant_target_small_error(dtype, value, num_samples):
    """Only an exact prediction receives a perfect score when the target is constant."""
    target = torch.full((num_samples,), value, dtype=dtype)
    for error, expected in [(0.0, 1.0), (1e-4, 0.0)]:
        preds = target + error
        assert r2_score(preds, target) == expected
        metric = R2Score().set_dtype(dtype)
        for pred_batch, target_batch in zip(preds.split(10), target.split(10)):
            metric.update(pred_batch, target_batch)
        assert metric.compute() == expected


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("multioutput", ["raw_values", "uniform_average", "variance_weighted"])
def test_r2_centered_state_merging(dtype, multioutput):
    """Forward accumulation and merging unequal batches preserve the full-data score."""
    target = torch.tensor([100.0, 101.0, 103.0, 105.0, 108.0, 113.0], dtype=dtype)
    target = target.unsqueeze(1) * torch.tensor([1e-3, 1.0], dtype=dtype)
    preds = target + torch.tensor([0.002, 1.0], dtype=dtype)
    expected = torch.as_tensor(sk_r2score(target.numpy(), preds.numpy(), multioutput=multioutput), dtype=dtype)
    first = R2Score(multioutput=multioutput).set_dtype(dtype)
    second = R2Score(multioutput=multioutput).set_dtype(dtype)
    first.update(preds[:2], target[:2])
    second.update(preds[2:], target[2:])
    first.merge_state(second)
    torch.testing.assert_close(first.compute(), expected, atol=1e-4, rtol=1e-4)

    metric = R2Score(multioutput=multioutput).set_dtype(dtype)
    metric(preds[:2], target[:2])
    metric(preds[2:], target[2:])
    torch.testing.assert_close(metric.compute(), expected, atol=1e-4, rtol=1e-4)

    empty = R2Score(multioutput=multioutput).set_dtype(dtype)
    empty.merge_state(metric)
    empty.merge_state(R2Score(multioutput=multioutput).set_dtype(dtype))
    torch.testing.assert_close(empty.compute(), expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("target", [torch.tensor([0.0, float("nan")]), torch.tensor([-1e20, 1e20])])
def test_r2_nonfinite_statistics(target):
    """Invalid or overflowing statistics must not become a finite score."""
    preds = torch.zeros_like(target)
    assert torch.isnan(r2_score(preds, target))
    assert torch.isnan(R2Score()(preds, target))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("num_prior", [0, 3])
def test_r2_load_state_and_continue(dtype, legacy, num_prior):
    """Current and legacy persisted states can resume accumulation, including from an empty state."""
    target = torch.arange(6, dtype=dtype) * 1e-3
    preds = target + 5e-4
    metric = R2Score().set_dtype(dtype)
    metric.persistent(True)
    metric.update(preds[:num_prior], target[:num_prior])
    state = metric.state_dict()
    if legacy:
        state.pop("target_mean")
        state.pop("target_sum_squared_deviation")
    restored = R2Score().set_dtype(dtype)
    restored.load_state_dict(state)
    restored.update(preds[num_prior:], target[num_prior:])
    expected = torch.as_tensor(sk_r2score(target.numpy(), preds.numpy()), dtype=dtype)
    torch.testing.assert_close(restored.compute(), expected, atol=1e-4, rtol=1e-4)
