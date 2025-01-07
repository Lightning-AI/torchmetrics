import numpy as np
import pytest
import torch

from torchmetrics.aggregation import CatMetric, MaxMetric, MeanMetric, MinMetric, SumMetric
from torchmetrics.collections import MetricCollection
from unittests import BATCH_SIZE, NUM_BATCHES
from unittests._helpers.testers import MetricTester


def compare_mean(values, weights):
    """Baseline implementation for mean aggregation."""
    return np.average(values.numpy(), weights=weights)


def compare_sum(values, weights):
    """Baseline implementation for sum aggregation."""
    return np.sum(values.numpy())


def compare_min(values, weights):
    """Baseline implementation for min aggregation."""
    return np.min(values.numpy())


def compare_max(values, weights):
    """Baseline implementation for max aggregation."""
    return np.max(values.numpy())


# wrap all other than mean metric to take an additional argument
# this lets them fit into the testing framework
class WrappedMinMetric(MinMetric):
    """Wrapped min metric."""

    def update(self, values, weights):
        """Only pass values on."""
        super().update(values)


class WrappedMaxMetric(MaxMetric):
    """Wrapped max metric."""

    def update(self, values, weights):
        """Only pass values on."""
        super().update(values)


class WrappedSumMetric(SumMetric):
    """Wrapped min metric."""

    def update(self, values, weights):
        """Only pass values on."""
        super().update(values)


class WrappedCatMetric(CatMetric):
    """Wrapped cat metric."""

    def update(self, values, weights):
        """Only pass values on."""
        super().update(values)


@pytest.mark.parametrize(
    "values, weights",
    [
        (torch.rand(NUM_BATCHES, BATCH_SIZE), torch.ones(NUM_BATCHES, BATCH_SIZE)),
        (torch.rand(NUM_BATCHES, BATCH_SIZE), torch.rand(NUM_BATCHES, BATCH_SIZE) > 0.5),
        (torch.rand(NUM_BATCHES, BATCH_SIZE, 2), torch.rand(NUM_BATCHES, BATCH_SIZE, 2) > 0.5),
    ],
)
@pytest.mark.parametrize(
    "metric_class, compare_fn",
    [
        (WrappedMinMetric, compare_min),
        (WrappedMaxMetric, compare_max),
        (WrappedSumMetric, compare_sum),
        (MeanMetric, compare_mean),
    ],
)
class TestAggregation(MetricTester):
    """Test aggregation metrics."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_aggreagation(self, ddp, metric_class, compare_fn, values, weights):
        """Test modular implementation."""
        self.run_class_metric_test(
            ddp=ddp,
            metric_class=metric_class,
            reference_metric=compare_fn,
            check_scriptable=True,
            # Abuse of names here
            preds=values,
            target=weights,
        )


_CASE_1 = float("nan") * torch.ones(5)
_CASE_2 = torch.tensor([1.0, 2.0, float("nan"), 4.0, 5.0])


@pytest.mark.parametrize("value", [_CASE_1, _CASE_2])
@pytest.mark.parametrize("nan_strategy", ["error", "warn"])
@pytest.mark.parametrize("metric_class", [MinMetric, MaxMetric, SumMetric, MeanMetric, CatMetric])
def test_nan_error(value, nan_strategy, metric_class):
    """Test correct errors are raised."""
    metric = metric_class(nan_strategy=nan_strategy)
    if nan_strategy == "error":
        with pytest.raises(RuntimeError, match="Encountered `nan` values in tensor"):
            metric(value.clone())
    elif nan_strategy == "warn":
        with pytest.warns(UserWarning, match="Encountered `nan` values in tensor"):
            metric(value.clone())


@pytest.mark.parametrize(
    ("metric_class", "nan_strategy", "value", "expected"),
    [
        (MinMetric, "ignore", _CASE_1, torch.tensor(float("inf"))),
        (MinMetric, 2.0, _CASE_1, 2.0),
        (MinMetric, "ignore", _CASE_2, 1.0),
        (MinMetric, 2.0, _CASE_2, 1.0),
        (MaxMetric, "ignore", _CASE_1, -torch.tensor(float("inf"))),
        (MaxMetric, 2.0, _CASE_1, 2.0),
        (MaxMetric, "ignore", _CASE_2, 5.0),
        (MaxMetric, 2.0, _CASE_2, 5.0),
        (SumMetric, "ignore", _CASE_1, 0.0),
        (SumMetric, 2.0, _CASE_1, 10.0),
        (SumMetric, "ignore", _CASE_2, 12.0),
        (SumMetric, 2.0, _CASE_2, 14.0),
        (MeanMetric, "ignore", _CASE_1, torch.tensor([float("nan")])),
        (MeanMetric, 2.0, _CASE_1, 2.0),
        (MeanMetric, "ignore", _CASE_2, 3.0),
        (MeanMetric, 2.0, _CASE_2, 2.8),
        (CatMetric, "ignore", _CASE_1, []),
        (CatMetric, 2.0, _CASE_1, torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0])),
        (CatMetric, "ignore", _CASE_2, torch.tensor([1.0, 2.0, 4.0, 5.0])),
        (CatMetric, 2.0, _CASE_2, torch.tensor([1.0, 2.0, 2.0, 4.0, 5.0])),
        (CatMetric, "ignore", torch.zeros(5), torch.zeros(5)),
    ],
)
def test_nan_expected(metric_class, nan_strategy, value, expected):
    """Test that nan values are handled correctly."""
    metric = metric_class(nan_strategy=nan_strategy)
    metric.update(value.clone())
    out = metric.compute()
    assert np.allclose(out, expected, equal_nan=True)


@pytest.mark.parametrize("metric_class", [MinMetric, MaxMetric, SumMetric, MeanMetric, CatMetric])
def test_error_on_wrong_nan_strategy(metric_class):
    """Test error raised on wrong nan_strategy argument."""
    with pytest.raises(ValueError, match="Arg `nan_strategy` should either .*"):
        metric_class(nan_strategy=[])


@pytest.mark.skipif(not hasattr(torch, "broadcast_to"), reason="PyTorch <1.8 does not have broadcast_to")
@pytest.mark.parametrize(
    ("weights", "expected"), [(1, 11.5), (torch.ones(2, 1, 1), 11.5), (torch.tensor([1, 2]).reshape(2, 1, 1), 13.5)]
)
def test_mean_metric_broadcasting(weights, expected):
    """Check that weight broadcasting works for mean metric."""
    values = torch.arange(24).reshape(2, 3, 4)
    avg = MeanMetric()

    assert avg(values, weights) == expected


def test_aggregation_in_collection_with_compute_groups():
    """Check that aggregation metrics work in MetricCollection with compute_groups=True."""
    m = MetricCollection(MinMetric(), MaxMetric(), SumMetric(), MeanMetric(), compute_groups=True)
    assert len(m.compute_groups) == 4, "Expected 4 compute groups"
    m.update(1)
    assert len(m.compute_groups) == 4, "Expected 4 compute groups"
    m.update(2)
    assert len(m.compute_groups) == 4, "Expected 4 compute groups"

    res = m.compute()
    assert res["MinMetric"] == 1
    assert res["MaxMetric"] == 2
    assert res["SumMetric"] == 3
    assert res["MeanMetric"] == 1.5


@pytest.mark.skipif(not hasattr(torch, "broadcast_to"), reason="PyTorch <1.8 does not have broadcast_to")
@pytest.mark.parametrize("nan_strategy", ["ignore", "warn"])
def test_mean_metric_broadcast(nan_strategy):
    """Check that weights gets broadcasted correctly when Nans are present."""
    metric = MeanMetric(nan_strategy=nan_strategy)

    x = torch.arange(5).float()
    x[1] = torch.tensor(float("nan"))
    w = torch.arange(5).float()

    metric.update(x, w)
    res = metric.compute()
    assert round(res.item(), 4) == 3.2222  # (0*0 + 2*2 + 3*3 + 4*4) / (0 + 2 + 3 + 4)

    x = torch.arange(5).float()
    w = torch.arange(5).float()
    w[1] = torch.tensor(float("nan"))

    metric.update(x, w)
    res = metric.compute()
    assert round(res.item(), 4) == 3.2222  # (0*0 + 2*2 + 3*3 + 4*4) / (0 + 2 + 3 + 4)


@pytest.mark.parametrize(
    ("metric_class", "compare_function"),
    [(MinMetric, torch.min), (MaxMetric, torch.max), (SumMetric, torch.sum), (MeanMetric, torch.mean)],
)
def test_with_default_dtype(metric_class, compare_function):
    """Test that the metric works with a default dtype of float64."""
    torch.set_default_dtype(torch.float64)
    metric = metric_class()
    assert metric.dtype == torch.float64
    values = torch.randn(10000)
    metric.update(values)
    result = metric.compute()
    assert result.dtype == torch.float64
    assert result.dtype == values.dtype
    assert result == compare_function(values)
    torch.set_default_dtype(torch.float32)
