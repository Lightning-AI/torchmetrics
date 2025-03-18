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
import sys
from functools import partial

import pytest
import torch

from torchmetrics.aggregation import MeanMetric, SumMetric
from torchmetrics.classification import BinaryAccuracy, BinaryConfusionMatrix
from torchmetrics.collections import MetricCollection
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, PearsonCorrCoef
from torchmetrics.wrappers import Running
from unittests import NUM_PROCESSES, USE_PYTEST_POOL


def test_errors_on_wrong_input():
    """Make sure that input type errors are raised on the wrong input."""
    with pytest.raises(ValueError, match="Expected argument `metric` to be an instance of `torchmetrics.Metric` .*"):
        Running(1)

    with pytest.raises(ValueError, match="Expected argument `window` to be a positive integer but got -1"):
        Running(SumMetric(), window=-1)

    with pytest.raises(ValueError, match="Expected attribute `full_state_update` set to `False` but got True"):
        Running(PearsonCorrCoef(), window=3)


def test_basic_aggregation():
    """Make sure that the aggregation works as expected for simple aggregate metrics."""
    metric = Running(SumMetric(), window=3)

    for i in range(10):
        metric.update(i)
        val = metric.compute()
        assert val == (i + max(i - 1, 0) + max(i - 2, 0)), f"Running sum is not correct in step {i}"

    metric = Running(MeanMetric(), window=3)

    for i in range(10):
        metric.update(i)
        val = metric.compute()
        assert val == (i + max(i - 1, 0) + max(i - 2, 0)) / min(i + 1, 3), f"Running mean is not correct in step {i}"


def test_forward():
    """Check that forward method works as expected."""
    compare_metric = SumMetric()
    metric = Running(SumMetric(), window=3)

    for i in range(10):
        assert compare_metric(i) == metric(i)
        assert metric.compute() == (i + max(i - 1, 0) + max(i - 2, 0)), f"Running sum is not correct in step {i}"

    compare_metric = MeanMetric()
    metric = Running(MeanMetric(), window=3)

    for i in range(10):
        assert compare_metric(i) == metric(i)
        assert metric.compute() == (i + max(i - 1, 0) + max(i - 2, 0)) / min(i + 1, 3), (
            f"Running mean is not correct in step {i}"
        )


@pytest.mark.parametrize(
    ("metric", "preds", "target"),
    [
        (BinaryAccuracy, torch.rand(10, 20), torch.randint(2, (10, 20))),
        (BinaryConfusionMatrix, torch.rand(10, 20), torch.randint(2, (10, 20))),
        (MeanSquaredError, torch.rand(10, 20), torch.rand(10, 20)),
        (MeanAbsoluteError, torch.rand(10, 20), torch.rand(10, 20)),
    ],
)
@pytest.mark.parametrize("window", [1, 3, 5])
def test_advance_running(metric, preds, target, window):
    """Check that running metrics work as expected for metrics that require advance computation."""
    base_metric = metric()
    running_metric = Running(metric(), window=window)

    for i in range(10):  # using forward
        p, t = preds[i], target[i]
        p_run = preds[max(i - (window - 1), 0) : i + 1, :].reshape(-1)
        t_run = target[max(i - (window - 1), 0) : i + 1, :].reshape(-1)

        assert torch.allclose(base_metric(p, t), running_metric(p, t))
        assert torch.allclose(base_metric(p_run, t_run), running_metric.compute())

    base_metric.reset()
    running_metric.reset()

    for i in range(10):  # using update
        p, t = preds[i], target[i]
        p_run, t_run = (
            preds[max(i - (window - 1), 0) : i + 1, :].reshape(-1),
            target[max(i - (window - 1), 0) : i + 1, :].reshape(-1),
        )

        running_metric.update(p, t)
        assert torch.allclose(base_metric(p_run, t_run), running_metric.compute())


@pytest.mark.parametrize("window", [3, 5])
def test_metric_collection(window):
    """Check that running metric works as expected for metric collections."""
    compare = MetricCollection({"mse": MeanSquaredError(), "msa": MeanAbsoluteError()})
    metric = MetricCollection({
        "mse": Running(MeanSquaredError(), window=window),
        "msa": Running(MeanAbsoluteError(), window=window),
    })
    preds = torch.rand(10, 20)
    target = torch.rand(10, 20)

    for i in range(10):
        p, t = preds[i], target[i]
        p_run, t_run = (
            preds[max(i - (window - 1), 0) : i + 1, :].reshape(-1),
            target[max(i - (window - 1), 0) : i + 1, :].reshape(-1),
        )
        metric.update(p, t)

        res1, res2 = compare(p_run, t_run), metric.compute()
        for key in res1:
            assert torch.allclose(res1[key], res2[key])


def _test_ddp_running(rank, dist_sync_on_step, expected):
    """Worker function for ddp test."""
    metric = Running(SumMetric(dist_sync_on_step=dist_sync_on_step), window=3)
    for _ in range(10):
        out = metric(torch.tensor(1.0))
        assert out == expected
    assert metric.compute() == 6


@pytest.mark.DDP
@pytest.mark.skipif(sys.platform == "win32", reason="DDP not available on windows")
@pytest.mark.skipif(not USE_PYTEST_POOL, reason="DDP pool is not available.")
@pytest.mark.parametrize(("dist_sync_on_step", "expected"), [(False, 1), (True, 2)])
def test_ddp_running(dist_sync_on_step, expected):
    """Check that the dist_sync_on_step gets correctly passed to base metric."""
    pytest.pool.map(
        partial(_test_ddp_running, dist_sync_on_step=dist_sync_on_step, expected=expected), range(NUM_PROCESSES)
    )


def test_no_warning_due_to_reset(recwarn):
    """Internally we call .reset() which would normally raise a warning, but it should not happen in Runner."""
    metric = Running(SumMetric(), window=3)
    metric.update(torch.tensor(2.0))
    assert len(recwarn) == 0, f"Warnings: {recwarn.list}"
