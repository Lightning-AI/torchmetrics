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
import math
from functools import partial

import numpy as np
import pytest
import torch

from torchmetrics.functional.regression.log_cosh import log_cosh_error
from torchmetrics.regression.log_cosh import LogCoshError
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


def _reference_log_cosh_error(preds, target):
    preds, target = preds.numpy(), target.numpy()
    diff = preds - target
    log_cosh = np.logaddexp(diff, -diff) - np.log(2)
    if diff.ndim == 1:
        return np.mean(log_cosh)
    return np.mean(log_cosh, axis=0)


@pytest.mark.parametrize(
    ("dtype", "value", "num_samples", "rtol"),
    [
        (torch.float16, 500.0, 1001, 1e-3),
        (torch.bfloat16, 100.0, 2, 5e-3),
        (torch.float32, 100.0, 2, 1e-6),
        (torch.float64, 1000.0, 2, 1e-6),
    ],
)
@pytest.mark.parametrize("metric_class", [None, LogCoshError], ids=["functional", "class"])
def test_log_cosh_error_large_residuals(metric_class, dtype, value, num_samples, rtol):
    """Test that large finite residuals produce a finite result."""
    preds = torch.full((num_samples,), value, dtype=dtype)
    preds[1::2] = -value
    target = torch.zeros_like(preds)

    result = log_cosh_error(preds, target) if metric_class is None else metric_class()(preds, target)
    expected = torch.tensor(value - math.log(2.0), dtype=result.dtype)

    assert torch.isfinite(result)
    torch.testing.assert_close(result, expected, rtol=rtol, atol=0)


@pytest.mark.parametrize("metric_class", [None, LogCoshError], ids=["functional", "class"])
def test_log_cosh_error_half_residual_exceeds_dtype_range(metric_class):
    """Test that subtraction is performed after promoting half-precision inputs."""
    value = torch.finfo(torch.float16).max
    preds = torch.tensor([value, -value], dtype=torch.float16)
    target = -preds

    result = log_cosh_error(preds, target) if metric_class is None else metric_class()(preds, target)
    expected = torch.tensor(2 * value - math.log(2.0), dtype=result.dtype)

    assert torch.isfinite(result)
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize(
    ("preds", "target"),
    [
        (_single_target_inputs.preds, _single_target_inputs.target),
        (_multi_target_inputs.preds, _multi_target_inputs.target),
    ],
)
class TestLogCoshError(MetricTester):
    """Test class for `LogCoshError` metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_log_cosh_error_class(self, ddp, preds, target):
        """Test class implementation of metric."""
        num_outputs = 1 if preds.ndim == 2 else NUM_TARGETS
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=LogCoshError,
            reference_metric=_reference_log_cosh_error,
            metric_args={"num_outputs": num_outputs},
        )

    def test_log_cosh_error_functional(self, preds, target):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=log_cosh_error,
            reference_metric=_reference_log_cosh_error,
        )

    def test_log_cosh_error_differentiability(self, preds, target):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        num_outputs = 1 if preds.ndim == 2 else NUM_TARGETS
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=partial(LogCoshError, num_outputs=num_outputs),
            metric_functional=log_cosh_error,
        )
