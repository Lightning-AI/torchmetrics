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

import numpy as np
import pytest
import torch
from torchmetrics.functional.regression.log_cosh import log_cosh_error
from torchmetrics.regression.log_cosh import LogCoshError

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


def _sk_log_cosh_error(preds, target):
    preds, target = preds.numpy(), target.numpy()
    diff = preds - target
    if diff.ndim == 1:
        return np.mean(np.log((np.exp(diff) + np.exp(-diff)) / 2))
    return np.mean(np.log((np.exp(diff) + np.exp(-diff)) / 2), axis=0)


@pytest.mark.parametrize(
    "preds, target",
    [
        (_single_target_inputs.preds, _single_target_inputs.target),
        (_multi_target_inputs.preds, _multi_target_inputs.target),
    ],
)
class TestLogCoshError(MetricTester):
    """Test class for `LogCoshError` metric."""

    @pytest.mark.parametrize("ddp", [True, False])
    def test_log_cosh_error_class(self, ddp, preds, target):
        """Test class implementation of metric."""
        num_outputs = 1 if preds.ndim == 2 else num_targets
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=LogCoshError,
            reference_metric=_sk_log_cosh_error,
            metric_args={"num_outputs": num_outputs},
        )

    def test_log_cosh_error_functional(self, preds, target):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=log_cosh_error,
            reference_metric=_sk_log_cosh_error,
        )

    def test_log_cosh_error_differentiability(self, preds, target):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        num_outputs = 1 if preds.ndim == 2 else num_targets
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=partial(LogCoshError, num_outputs=num_outputs),
            metric_functional=log_cosh_error,
        )
