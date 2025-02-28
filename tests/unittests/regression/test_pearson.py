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
from scipy.stats import pearsonr

from torchmetrics.functional.regression.pearson import pearson_corrcoef
from torchmetrics.regression.pearson import PearsonCorrCoef, _final_aggregation
from unittests import BATCH_SIZE, EXTRA_DIM, NUM_BATCHES, _Input
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

seed_all(42)


_single_target_inputs1 = _Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE),
)

_single_target_inputs2 = _Input(
    preds=torch.randn(NUM_BATCHES, BATCH_SIZE),
    target=torch.randn(NUM_BATCHES, BATCH_SIZE),
)


_multi_target_inputs1 = _Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM),
)

_multi_target_inputs2 = _Input(
    preds=torch.randn(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM),
    target=torch.randn(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM),
)


def _reference_scipy_pearson(preds, target):
    if preds.ndim == 2:
        return [pearsonr(t.numpy(), p.numpy())[0] for t, p in zip(target.T, preds.T)]
    return pearsonr(target.numpy(), preds.numpy())[0]


@pytest.mark.parametrize(
    "preds, target",
    [
        (_single_target_inputs1.preds, _single_target_inputs1.target),
        (_single_target_inputs2.preds, _single_target_inputs2.target),
        (_multi_target_inputs1.preds, _multi_target_inputs1.target),
        (_multi_target_inputs2.preds, _multi_target_inputs2.target),
    ],
)
class TestPearsonCorrCoef(MetricTester):
    """Test class for `PearsonCorrCoef` metric."""

    atol = 1e-3

    @pytest.mark.parametrize("compute_on_cpu", [True, False])
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_pearson_corrcoef(self, preds, target, compute_on_cpu, ddp):
        """Test class implementation of metric."""
        num_outputs = EXTRA_DIM if preds.ndim == 3 else 1
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=PearsonCorrCoef,
            reference_metric=_reference_scipy_pearson,
            metric_args={"num_outputs": num_outputs, "compute_on_cpu": compute_on_cpu},
        )

    def test_pearson_corrcoef_functional(self, preds, target):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds, target=target, metric_functional=pearson_corrcoef, reference_metric=_reference_scipy_pearson
        )

    def test_pearson_corrcoef_differentiability(self, preds, target):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        num_outputs = EXTRA_DIM if preds.ndim == 3 else 1
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=partial(PearsonCorrCoef, num_outputs=num_outputs),
            metric_functional=pearson_corrcoef,
        )

    def test_pearson_corrcoef_half_cpu(self, preds, target):
        """Test dtype support of the metric on CPU."""
        num_outputs = EXTRA_DIM if preds.ndim == 3 else 1
        self.run_precision_test_cpu(preds, target, partial(PearsonCorrCoef, num_outputs=num_outputs), pearson_corrcoef)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_pearson_corrcoef_half_gpu(self, preds, target):
        """Test dtype support of the metric on GPU."""
        num_outputs = EXTRA_DIM if preds.ndim == 3 else 1
        self.run_precision_test_gpu(preds, target, partial(PearsonCorrCoef, num_outputs=num_outputs), pearson_corrcoef)


def test_error_on_different_shape():
    """Test that error is raised on different shapes of input."""
    metric = PearsonCorrCoef(num_outputs=1)
    with pytest.raises(RuntimeError, match="Predictions and targets are expected to have the same shape"):
        metric(torch.randn(100), torch.randn(50))

    metric = PearsonCorrCoef(num_outputs=5)
    with pytest.raises(ValueError, match="Expected both predictions and target to be either 1- or 2-.*"):
        metric(torch.randn(100, 2, 5), torch.randn(100, 2, 5))

    metric = PearsonCorrCoef(num_outputs=2)
    with pytest.raises(ValueError, match="Expected argument `num_outputs` to match the second dimension of input.*"):
        metric(torch.randn(100, 5), torch.randn(100, 5))


def test_1d_input_allowed():
    """Check that both input of the form [N,] and [N,1] is allowed with default num_outputs argument."""
    assert isinstance(pearson_corrcoef(torch.randn(10, 1), torch.randn(10, 1)), torch.Tensor)
    assert isinstance(pearson_corrcoef(torch.randn(10), torch.randn(10)), torch.Tensor)


@pytest.mark.parametrize("shapes", [(5,), (1, 5), (2, 5)])
def test_final_aggregation_function(shapes):
    """Test that final aggregation function can take various shapes of input."""
    input_fn = lambda: torch.rand(shapes)
    output = _final_aggregation(input_fn(), input_fn(), input_fn(), input_fn(), input_fn(), torch.randint(10, shapes))
    assert all(isinstance(out, torch.Tensor) for out in output)
    assert all(out.ndim == input_fn().ndim - 1 for out in output)


@pytest.mark.parametrize(("dtype", "scale"), [(torch.float16, 1e-4), (torch.float32, 1e-8), (torch.float64, 1e-16)])
def test_pearsons_warning_on_small_input(dtype, scale):
    """Check that a user warning is raised for small input."""
    preds = scale * torch.randn(100, dtype=dtype)
    target = scale * torch.randn(100, dtype=dtype)
    with pytest.warns(UserWarning, match="The variance of predictions or target is close to zero.*"):
        pearson_corrcoef(preds, target)


def test_single_sample_update():
    """See issue: https://github.com/Lightning-AI/torchmetrics/issues/2014."""
    metric = PearsonCorrCoef()

    # Works
    metric(torch.tensor([3.0, -0.5, 2.0, 7.0]), torch.tensor([2.5, 0.0, 2.0, 8.0]))
    res1 = metric.compute()
    metric.reset()

    metric(torch.tensor([3.0]), torch.tensor([2.5]))
    metric(torch.tensor([-0.5]), torch.tensor([0.0]))
    metric(torch.tensor([2.0]), torch.tensor([2.0]))
    metric(torch.tensor([7.0]), torch.tensor([8.0]))
    res2 = metric.compute()
    assert torch.allclose(res1, res2)


def test_overwrite_reference_inputs():
    """Test that the normalizations does not overwrite inputs.

    Variables var_x, var_y, corr_xy are references to the object variables and get incorrectly scaled down such that
    when you update again and compute you get very wrong values.

    """
    y = torch.randn(100)
    y_pred = y + torch.randn(y.shape) / 5
    # Initialize Pearson correlation coefficient metric
    pearson = PearsonCorrCoef()
    # Compute the Pearson correlation coefficient
    correlation = pearson(y, y_pred)

    pearson = PearsonCorrCoef()
    for lower, upper in [(0, 33), (33, 66), (66, 99), (99, 100)]:
        pearson.update(torch.tensor(y[lower:upper]), torch.tensor(y_pred[lower:upper]))
        pearson.compute()

    assert torch.isclose(pearson.compute(), correlation)
