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
from torchmetrics.functional.regression.weighted_pearson import weighted_pearson_corrcoef
from torchmetrics.regression.pearson import PearsonCorrCoef, _final_aggregation
from torchmetrics.regression.weighted_pearson import WeightedPearsonCorrCoef
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_2_5
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

_weights = torch.rand(NUM_BATCHES, BATCH_SIZE)


def _reference_scipy_pearson(preds, target):
    if preds.ndim == 2:
        return [pearsonr(t.numpy(), p.numpy())[0] for t, p in zip(target.T, preds.T)]
    return pearsonr(target.numpy(), preds.numpy())[0]


def _reference_weighted_pearson(preds, target, weights):
    if preds.ndim == 2:
        return [_reference_weighted_pearson(p, t, weights) for p, t in zip(preds.T, target.T)]

    mx = (weights * preds).sum() / weights.sum()
    my = (weights * target).sum() / weights.sum()
    var_x = (weights * (preds - mx) ** 2).sum()
    var_y = (weights * (target - my) ** 2).sum()
    cov_xy = (weights * (preds - mx) * (target - my)).sum()
    return cov_xy / (var_x * var_y).sqrt()


@pytest.mark.parametrize(
    ("preds", "target"),
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

    @pytest.mark.skipif(not _TORCH_GREATER_EQUAL_2_5, reason="Requires torch>=2.5.0")
    def test_pearson_corrcoef_half_cpu(self, preds, target):
        """Test dtype support of the metric on CPU."""
        num_outputs = EXTRA_DIM if preds.ndim == 3 else 1
        self.run_precision_test_cpu(preds, target, partial(PearsonCorrCoef, num_outputs=num_outputs), pearson_corrcoef)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_pearson_corrcoef_half_gpu(self, preds, target):
        """Test dtype support of the metric on GPU."""
        num_outputs = EXTRA_DIM if preds.ndim == 3 else 1
        self.run_precision_test_gpu(preds, target, partial(PearsonCorrCoef, num_outputs=num_outputs), pearson_corrcoef)


@pytest.mark.parametrize(
    ("preds", "target", "weights"),
    [
        (_single_target_inputs1.preds, _single_target_inputs1.target, _weights),
        (_single_target_inputs2.preds, _single_target_inputs2.target, _weights),
        (_multi_target_inputs1.preds, _multi_target_inputs1.target, _weights),
        (_multi_target_inputs2.preds, _multi_target_inputs2.target, _weights),
    ],
)
class TestWeightedPearsonCorrCoef(MetricTester):
    """Test class for `WeightedPearsonCorrCoef` metric."""

    atol = 1e-3

    @pytest.mark.parametrize("compute_on_cpu", [True, False])
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_pearson_corrcoef(self, preds, target, weights, compute_on_cpu, ddp):
        """Test class implementation of metric."""
        num_outputs = EXTRA_DIM if preds.ndim == 3 else 1
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=WeightedPearsonCorrCoef,
            reference_metric=_reference_weighted_pearson,
            metric_args={"num_outputs": num_outputs, "compute_on_cpu": compute_on_cpu},
            weights=weights,
        )

    def test_pearson_corrcoef_functional(self, preds, target, weights):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=weighted_pearson_corrcoef,
            reference_metric=_reference_weighted_pearson,
            weights=weights,
        )

    def test_pearson_corrcoef_differentiability(self, preds, target, weights):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        num_outputs = EXTRA_DIM if preds.ndim == 3 else 1
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=partial(WeightedPearsonCorrCoef, num_outputs=num_outputs),
            metric_functional=weighted_pearson_corrcoef,
            weights=weights,
        )

    @pytest.mark.skipif(not _TORCH_GREATER_EQUAL_2_5, reason="Requires torch>=2.5.0")
    def test_pearson_corrcoef_half_cpu(self, preds, target, weights):
        """Test dtype support of the metric on CPU."""
        num_outputs = EXTRA_DIM if preds.ndim == 3 else 1
        self.run_precision_test_cpu(
            preds,
            target,
            partial(WeightedPearsonCorrCoef, num_outputs=num_outputs),
            weighted_pearson_corrcoef,
            weights=weights,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_pearson_corrcoef_half_gpu(self, preds, target, weights):
        """Test dtype support of the metric on GPU."""
        num_outputs = EXTRA_DIM if preds.ndim == 3 else 1
        self.run_precision_test_gpu(
            preds,
            target,
            partial(WeightedPearsonCorrCoef, num_outputs=num_outputs),
            weighted_pearson_corrcoef,
            weights=weights,
        )


@pytest.mark.parametrize(
    ("metric_class", "metric_args"),
    [
        (PearsonCorrCoef, [torch.randn(100), torch.randn(50)]),
        (WeightedPearsonCorrCoef, [torch.randn(100), torch.randn(50), torch.randn(50)]),
    ],
)
def test_error_on_different_shape(metric_class, metric_args):
    """Test that error is raised on different shapes of input."""
    metric = metric_class(num_outputs=1)
    with pytest.raises(RuntimeError, match="Predictions and targets are expected to have the same shape"):
        metric(*metric_args)


@pytest.mark.parametrize(
    ("metric_class", "metric_args"),
    [
        (PearsonCorrCoef, [torch.randn(100, 2, 5), torch.randn(100, 2, 5)]),
        (WeightedPearsonCorrCoef, [torch.randn(100, 2, 5), torch.randn(100, 2, 5), torch.randn(100)]),
    ],
)
def test_error_on_invalid_ndim(metric_class, metric_args):
    """Test that error is raised on invalid dimensions."""
    metric = metric_class(num_outputs=5)
    with pytest.raises(ValueError, match="Expected both predictions and target to be either 1- or 2-.*"):
        metric(*metric_args)


@pytest.mark.parametrize(
    ("metric_class", "metric_args"),
    [
        (PearsonCorrCoef, [torch.randn(100, 3), torch.randn(100, 3)]),
        (WeightedPearsonCorrCoef, [torch.randn(100, 3), torch.randn(100, 3), torch.randn(100)]),
    ],
)
def test_error_on_num_outputs_mismatch(metric_class, metric_args):
    """Test that error is raised if `num_outputs` of `preds` or `target` do not match initialization."""
    metric = metric_class(num_outputs=2)
    with pytest.raises(ValueError, match="Expected argument `num_outputs` to match the second dimension of input.*"):
        metric(*metric_args)


@pytest.mark.parametrize(
    ("metric_functional", "metric_args"),
    [
        (pearson_corrcoef, [[torch.randn(10, 1), torch.randn(10, 1)], [torch.randn(10), torch.randn(10)]]),
        (
            weighted_pearson_corrcoef,
            [
                [torch.randn(10, 1), torch.randn(10, 1), torch.randn(10)],
                [torch.randn(10), torch.randn(10), torch.randn(10)],
            ],
        ),
    ],
)
def test_1d_input_allowed(metric_functional, metric_args):
    """Check that both input of the form [N,] and [N,1] is allowed with default num_outputs argument."""
    assert isinstance(metric_functional(*metric_args[0]), torch.Tensor)
    assert isinstance(metric_functional(*metric_args[1]), torch.Tensor)


@pytest.mark.parametrize("shapes", [(5,), (1, 5), (2, 5)])
def test_final_aggregation_function(shapes):
    """Test that final aggregation function can take various shapes of input."""
    input_fn = lambda: torch.rand(shapes)
    output = _final_aggregation(
        input_fn(), input_fn(), input_fn(), input_fn(), input_fn(), input_fn(), input_fn(), torch.randint(10, shapes)
    )
    assert all(isinstance(out, torch.Tensor) for out in output)
    assert all(out.ndim == input_fn().ndim - 1 for out in output)


def test_final_aggregation_no_inplace_change():
    """Test that final aggregation function does not change the input tensors in place."""
    n_devices = 2
    n_outputs = 100
    n_repeats = 2

    mean_x = torch.randn(n_devices, n_outputs)
    mean_y = torch.randn(n_devices, n_outputs)
    max_abs_dev_x = torch.randn(n_devices, n_outputs)
    max_abs_dev_y = torch.randn(n_devices, n_outputs)
    var_x = torch.randn(n_devices, n_outputs)
    var_y = torch.randn(n_devices, n_outputs)
    corr_xy = torch.randn(n_devices, n_outputs)
    n_total = torch.randint(1, 100, (n_devices, n_outputs))

    _mean_x = mean_x.clone()
    _mean_y = mean_y.clone()
    _max_abs_dev_x = max_abs_dev_x.clone()
    _max_abs_dev_y = max_abs_dev_y.clone()
    _var_x = var_x.clone()
    _var_y = var_y.clone()
    _corr_xy = corr_xy.clone()
    _n_total = n_total.clone()

    for _ in range(n_repeats):
        _final_aggregation(_mean_x, _mean_y, max_abs_dev_x, max_abs_dev_y, _var_x, _var_y, _corr_xy, _n_total)

    assert torch.allclose(_mean_x, mean_x), f"Mean X drift: mean={(_mean_x - mean_x).abs().mean().item()}"
    assert torch.allclose(_mean_y, mean_y), f"Mean Y drift: mean={(_mean_y - mean_y).abs().mean().item()}"
    assert torch.allclose(_max_abs_dev_x, max_abs_dev_x), (
        f"Max Abs X drift: mean={(_max_abs_dev_x - max_abs_dev_x).abs().mean().item()}"
    )
    assert torch.allclose(_max_abs_dev_y, max_abs_dev_y), (
        f"Max Abs Y drift: mean={(_max_abs_dev_y - max_abs_dev_y).abs().mean().item()}"
    )
    assert torch.allclose(_var_x, var_x), f"Var X drift: mean={(_var_x - var_x).abs().mean().item()}"
    assert torch.allclose(_var_y, var_y), f"Var Y drift: mean={(_var_y - var_y).abs().mean().item()}"
    assert torch.allclose(_corr_xy, corr_xy), f"Corr XY drift: mean={(_corr_xy - corr_xy).abs().mean().item()}"
    assert torch.allclose(_n_total, n_total), f"N Total drift: mean={(_n_total - n_total).abs().mean().item()}"


def test_final_aggregation_with_empty_devices():
    """Test that final aggregation function can handle the case where some devices have no data."""
    n_devices = 4
    n_outputs = 5
    mean_x = torch.randn(n_devices, n_outputs)
    mean_y = torch.randn(n_devices, n_outputs)
    max_abs_dev_x = torch.randn(n_devices, n_outputs).abs()
    max_abs_dev_y = torch.randn(n_devices, n_outputs).abs()
    var_x = torch.randn(n_devices, n_outputs).abs()
    var_y = torch.randn(n_devices, n_outputs).abs()
    corr_xy = torch.randn(n_devices, n_outputs)
    n_total = torch.randint(1, 100, (n_devices, n_outputs))

    for x in [mean_x, mean_y, max_abs_dev_x, max_abs_dev_y, var_x, var_y, corr_xy, n_total]:
        x[:2] = 0

    # Current
    mean_x_cur, mean_y_cur, max_abs_dev_x_cur, max_abs_dev_y_cur, var_x_cur, var_y_cur, corr_xy_cur, n_total_cur = (
        _final_aggregation(mean_x, mean_y, max_abs_dev_x, max_abs_dev_y, var_x, var_y, corr_xy, n_total)
    )
    # Expected
    mean_x_exp, mean_y_exp, max_abs_dev_x_exp, max_abs_dev_y_exp, var_x_exp, var_y_exp, corr_xy_exp, n_total_exp = (
        _final_aggregation(
            mean_x[2:], mean_y[2:], max_abs_dev_x[2:], max_abs_dev_y[2:], var_x[2:], var_y[2:], corr_xy[2:], n_total[2:]
        )
    )

    assert torch.allclose(mean_x_cur, mean_x_exp), f"mean_x: {mean_x_cur} (expected: {mean_x_exp})"
    assert torch.allclose(mean_y_cur, mean_y_exp), f"mean_y: {mean_y_cur} (expected: {mean_y_exp})"
    assert torch.allclose(max_abs_dev_x_cur, max_abs_dev_x_exp), (
        f"max_abs_dev_x: {max_abs_dev_x_cur} (expected: {max_abs_dev_x_exp})"
    )
    assert torch.allclose(max_abs_dev_y_cur, max_abs_dev_y_exp), (
        f"max_abs_dev_y: {max_abs_dev_y_cur} (expected: {max_abs_dev_y_exp})"
    )
    assert torch.allclose(var_x_cur, var_x_exp), f"var_x: {var_x_cur} (expected: {var_x_exp})"
    assert torch.allclose(var_y_cur, var_y_exp), f"var_y: {var_y_cur} (expected: {var_y_exp})"
    assert torch.allclose(corr_xy_cur, corr_xy_exp), f"corr_xy: {corr_xy_cur} (expected: {corr_xy_exp})"
    assert torch.allclose(n_total_cur, n_total_exp), f"n_total: {n_total_cur} (expected: {n_total_exp})"


@pytest.mark.parametrize(("dtype", "scale"), [(torch.float16, 1e-4), (torch.float32, 1e-32), (torch.float64, 1e-256)])
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


def test_corner_cases():
    """Test corner cases with zero variances.

    See issue: https://github.com/Lightning-AI/torchmetrics/issues/2920

    """
    y_pred = torch.tensor([[-0.1816, 0.6568, 0.9788, -0.1425], [-0.4111, 0.3940, 1.4834, 0.1322]])
    y_true = torch.tensor([[4.0268, 5.9401, 1.0000, 1.0000], [6.4956, 5.6684, 1.0000, 1.0000]])
    pearson_corr = PearsonCorrCoef(num_outputs=4)
    result = pearson_corr(y_pred, y_true)
    assert torch.allclose(result, torch.tensor([-1.0, 1.0, float("nan"), float("nan")]), equal_nan=True)
