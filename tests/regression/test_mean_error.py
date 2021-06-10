# Copyright The PyTorch Lightning team.
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
from collections import namedtuple
from functools import partial

import pytest
import torch
from sklearn.metrics import mean_absolute_error as sk_mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error as sk_mean_abs_percentage_error
from sklearn.metrics import mean_squared_error as sk_mean_squared_error
from sklearn.metrics import mean_squared_log_error as sk_mean_squared_log_error

from tests.helpers import seed_all
from tests.helpers.testers import BATCH_SIZE, NUM_BATCHES, MetricTester
from torchmetrics.functional import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_log_error,
)
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    MeanSquaredLogError,
)
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_1_6

seed_all(42)

num_targets = 5

Input = namedtuple('Input', ["preds", "target"])

_single_target_inputs = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE),
)

_multi_target_inputs = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, num_targets),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE, num_targets),
)


def _single_target_sk_metric(preds, target, sk_fn, metric_args):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()

    # `sk_target` and `sk_preds` switched to fix failing tests.
    # For more info, check https://github.com/PyTorchLightning/metrics/pull/248#issuecomment-841232277
    res = sk_fn(sk_target, sk_preds)

    return math.sqrt(res) if (metric_args and not metric_args['squared']) else res


def _multi_target_sk_metric(preds, target, sk_fn, metric_args):
    sk_preds = preds.view(-1, num_targets).numpy()
    sk_target = target.view(-1, num_targets).numpy()

    # `sk_target` and `sk_preds` switched to fix failing tests.
    # For more info, check https://github.com/PyTorchLightning/metrics/pull/248#issuecomment-841232277
    res = sk_fn(sk_target, sk_preds)

    return math.sqrt(res) if (metric_args and not metric_args['squared']) else res


@pytest.mark.parametrize(
    "preds, target, sk_metric",
    [
        (_single_target_inputs.preds, _single_target_inputs.target, _single_target_sk_metric),
        (_multi_target_inputs.preds, _multi_target_inputs.target, _multi_target_sk_metric),
    ],
)
@pytest.mark.parametrize(
    "metric_class, metric_functional, sk_fn, metric_args",
    [
        (MeanSquaredError, mean_squared_error, sk_mean_squared_error, {
            'squared': True
        }),
        (MeanSquaredError, mean_squared_error, sk_mean_squared_error, {
            'squared': False
        }),
        (MeanAbsoluteError, mean_absolute_error, sk_mean_absolute_error, {}),
        (MeanAbsolutePercentageError, mean_absolute_percentage_error, sk_mean_abs_percentage_error, {}),
        (MeanSquaredLogError, mean_squared_log_error, sk_mean_squared_log_error, {}),
    ],
)
class TestMeanError(MetricTester):

    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_mean_error_class(
        self, preds, target, sk_metric, metric_class, metric_functional, sk_fn, metric_args, ddp, dist_sync_on_step
    ):
        # todo: `metric_functional` is unused
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=metric_class,
            sk_metric=partial(sk_metric, sk_fn=sk_fn, metric_args=metric_args),
            dist_sync_on_step=dist_sync_on_step,
            metric_args=metric_args
        )

    def test_mean_error_functional(self, preds, target, sk_metric, metric_class, metric_functional, sk_fn, metric_args):
        # todo: `metric_class` is unused
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=metric_functional,
            sk_metric=partial(sk_metric, sk_fn=sk_fn, metric_args=metric_args),
            metric_args=metric_args
        )

    def test_mean_error_differentiability(
        self, preds, target, sk_metric, metric_class, metric_functional, sk_fn, metric_args
    ):
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=metric_class,
            metric_functional=metric_functional,
            metric_args=metric_args
        )

    @pytest.mark.skipif(
        not _TORCH_GREATER_EQUAL_1_6, reason='half support of core operations on not support before pytorch v1.6'
    )
    def test_mean_error_half_cpu(self, preds, target, sk_metric, metric_class, metric_functional, sk_fn, metric_args):
        if metric_class == MeanSquaredLogError:
            # MeanSquaredLogError half + cpu does not work due to missing support in torch.log
            pytest.xfail("MeanSquaredLogError metric does not support cpu + half precision")

        if metric_class == MeanAbsolutePercentageError:
            # MeanSquaredPercentageError half + cpu does not work due to missing support in torch.log
            pytest.xfail("MeanSquaredPercentageError metric does not support cpu + half precision")

        self.run_precision_test_cpu(preds, target, metric_class, metric_functional)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='test requires cuda')
    def test_mean_error_half_gpu(self, preds, target, sk_metric, metric_class, metric_functional, sk_fn, metric_args):
        self.run_precision_test_gpu(preds, target, metric_class, metric_functional)


@pytest.mark.parametrize(
    "metric_class", [MeanSquaredError, MeanAbsoluteError, MeanSquaredLogError, MeanAbsolutePercentageError]
)
def test_error_on_different_shape(metric_class):
    metric = metric_class()
    with pytest.raises(RuntimeError, match='Predictions and targets are expected to have the same shape'):
        metric(torch.randn(100, ), torch.randn(50, ))
