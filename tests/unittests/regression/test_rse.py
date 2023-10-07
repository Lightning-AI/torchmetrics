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
from collections import namedtuple
from functools import partial

import numpy as np
import pytest
import torch
from torchmetrics.functional import relative_squared_error
from torchmetrics.regression import RelativeSquaredError
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_2_1

from unittests import BATCH_SIZE, NUM_BATCHES
from unittests.helpers import seed_all
from unittests.helpers.testers import MetricTester

seed_all(42)

num_targets = 5

Input = namedtuple("Input", ["preds", "target"])

_single_target_inputs = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE),
)

_multi_target_inputs = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, num_targets),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE, num_targets),
)


def _sk_rse(target, preds, squared):
    mean = np.mean(target, axis=0, keepdims=True)
    error = target - preds
    sum_squared_error = np.sum(error * error, axis=0)
    deviation = target - mean
    sum_squared_deviation = np.sum(deviation * deviation, axis=0)
    rse = sum_squared_error / np.maximum(sum_squared_deviation, 1.17e-06)
    if not squared:
        rse = np.sqrt(rse)
    return np.mean(rse)


def _single_target_ref_metric(preds, target, squared):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()
    return _sk_rse(sk_target, sk_preds, squared=squared)


def _multi_target_ref_metric(preds, target, squared):
    sk_preds = preds.view(-1, num_targets).numpy()
    sk_target = target.view(-1, num_targets).numpy()
    return _sk_rse(sk_target, sk_preds, squared=squared)


@pytest.mark.parametrize("squared", [False, True])
@pytest.mark.parametrize(
    "preds, target, ref_metric, num_outputs",
    [
        (_single_target_inputs.preds, _single_target_inputs.target, _single_target_ref_metric, 1),
        (_multi_target_inputs.preds, _multi_target_inputs.target, _multi_target_ref_metric, num_targets),
    ],
)
class TestRelativeSquaredError(MetricTester):
    """Test class for `RelativeSquaredError` metric."""

    @pytest.mark.parametrize("ddp", [True, False])
    def test_rse(self, squared, preds, target, ref_metric, num_outputs, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            RelativeSquaredError,
            partial(ref_metric, squared=squared),
            metric_args={"squared": squared, "num_outputs": num_outputs},
        )

    def test_rse_functional(self, squared, preds, target, ref_metric, num_outputs):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            relative_squared_error,
            partial(ref_metric, squared=squared),
            metric_args={"squared": squared},
        )

    def test_rse_differentiability(self, squared, preds, target, ref_metric, num_outputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=partial(RelativeSquaredError, num_outputs=num_outputs),
            metric_functional=relative_squared_error,
            metric_args={"squared": squared},
        )

    @pytest.mark.skipif(
        not _TORCH_GREATER_EQUAL_2_1,
        reason="Pytoch below 2.1 does not support cpu + half precision used in `clamp_min_cpu`",
    )
    def test_rse_half_cpu(self, squared, preds, target, ref_metric, num_outputs):
        """Test dtype support of the metric on CPU."""
        self.run_precision_test_cpu(
            preds,
            target,
            partial(RelativeSquaredError, num_outputs=num_outputs),
            relative_squared_error,
            {"squared": squared},
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_rse_half_gpu(self, squared, preds, target, ref_metric, num_outputs):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(
            preds,
            target,
            partial(RelativeSquaredError, num_outputs=num_outputs),
            relative_squared_error,
            {"squared": squared},
        )


def test_error_on_different_shape(metric_class=RelativeSquaredError):
    """Test that error is raised on different shapes of input."""
    metric = metric_class()
    with pytest.raises(RuntimeError, match="Predictions and targets are expected to have the same shape"):
        metric(torch.randn(100), torch.randn(50))


def test_error_on_multidim_tensors(metric_class=RelativeSquaredError):
    """Test that error is raised if a larger than 2D tensor is given as input."""
    metric = metric_class()
    with pytest.raises(
        ValueError,
        match=r"Expected both prediction and target to be 1D or 2D tensors, but received tensors with dimension .",
    ):
        metric(torch.randn(10, 20, 5), torch.randn(10, 20, 5))
