from collections import namedtuple
from functools import partial

import numpy as np
import pytest
import torch
from scipy.spatial.distance import minkowski as scipy_minkowski
from sklearn.metrics import mean_squared_error

from torchmetrics.functional import minkowski_distance
from torchmetrics.regression import MinkowskiDistance
from torchmetrics.utilities.exceptions import TorchMetricsUserError
from unittests.helpers import seed_all
from unittests.helpers.testers import BATCH_SIZE, NUM_BATCHES, MetricTester

seed_all(42)

num_targets = 5

Input = namedtuple("Input", ["preds", "target"])

_single_target_inputs = Input(preds=torch.rand(NUM_BATCHES, BATCH_SIZE), target=torch.rand(NUM_BATCHES, BATCH_SIZE))

_multi_target_inputs = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, num_targets), target=torch.rand(NUM_BATCHES, BATCH_SIZE, num_targets)
)


def root_mean_squared_error(preds, target):
    preds, target = preds.numpy(), target.numpy()
    return np.sqrt(mean_squared_error(preds - target))


def _single_target_sk_metric(preds, target, sk_fn, metric_args):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()

    res = sk_fn(sk_preds, sk_target, p=metric_args["p"])

    return res


def _multi_target_sk_metric(preds, target, sk_fn, metric_args):
    sk_preds = preds.view(-1, num_targets).numpy()
    sk_target = target.view(-1, num_targets).numpy

    res = sk_fn(sk_preds, sk_target, p=metric_args["p"])

    return res


@pytest.mark.parametrize(
    "preds, target, sk_metric",
    [
        (_single_target_inputs.preds, _single_target_inputs.target, _single_target_sk_metric),
    ],
)
@pytest.mark.parametrize(
    "metric_class, metric_functional, sk_fn, metric_args",
    [
        (MinkowskiDistance, minkowski_distance, scipy_minkowski, {"p": 1}),
        (MinkowskiDistance, minkowski_distance, scipy_minkowski, {"p": 2}),
        (MinkowskiDistance, minkowski_distance, scipy_minkowski, {"p": 3}),
        (MinkowskiDistance, minkowski_distance, scipy_minkowski, {"p": 4}),
        (MinkowskiDistance, minkowski_distance, scipy_minkowski, {"p": 5}),
        (MinkowskiDistance, minkowski_distance, scipy_minkowski, {"p": 0.5}),
        (MinkowskiDistance, minkowski_distance, scipy_minkowski, {"p": 1.5}),
        (MinkowskiDistance, minkowski_distance, scipy_minkowski, {"p": -1.25}),
        (MinkowskiDistance, minkowski_distance, scipy_minkowski, {"p": -0.5}),
        (MinkowskiDistance, minkowski_distance, scipy_minkowski, {"p": -8}),
    ],
)
class TestMinkowskiDistance(MetricTester):
    @pytest.mark.parametrize("ddp", [False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_minkowski_distance_class(
        self, preds, target, sk_metric, metric_class, metric_functional, sk_fn, metric_args, ddp, dist_sync_on_step
    ):
        if metric_args["p"] < 0:
            pytest.xfail("p-value must not be less than 0")

        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=metric_class,
            sk_metric=partial(sk_metric, sk_fn=sk_fn, metric_args=metric_args),
            dist_sync_on_step=dist_sync_on_step,
            metric_args=metric_args,
        )

    def test_minkowski_distance_functional(
        self, preds, target, sk_metric, metric_class, metric_functional, sk_fn, metric_args
    ):
        if metric_args["p"] < 0:
            pytest.xfail("p-value must not be less than 0")

        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=metric_functional,
            sk_metric=partial(sk_metric, sk_fn=sk_fn, metric_args=metric_args),
            metric_args=metric_args,
        )
