from functools import partial

import pytest
import torch
from scipy.spatial.distance import minkowski as scipy_minkowski
from torchmetrics.functional import minkowski_distance
from torchmetrics.regression import MinkowskiDistance
from torchmetrics.utilities.exceptions import TorchMetricsUserError

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


def _sk_metric_single_target(preds, target, p):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()
    return scipy_minkowski(sk_preds, sk_target, p=p)


def _sk_metric_multi_target(preds, target, p):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()
    return scipy_minkowski(sk_preds, sk_target, p=p)


@pytest.mark.parametrize(
    "preds, target, ref_metric",
    [
        (_single_target_inputs.preds, _single_target_inputs.target, _sk_metric_single_target),
        (_multi_target_inputs.preds, _multi_target_inputs.target, _sk_metric_multi_target),
    ],
)
@pytest.mark.parametrize("p", [1, 2, 4, 1.5])
class TestMinkowskiDistance(MetricTester):
    """Test class for `MinkowskiDistance` metric."""

    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_minkowski_distance_class(self, preds, target, ref_metric, p, ddp, dist_sync_on_step):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MinkowskiDistance,
            reference_metric=partial(ref_metric, p=p),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={"p": p},
        )

    def test_minkowski_distance_functional(self, preds, target, ref_metric, p):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=minkowski_distance,
            reference_metric=partial(ref_metric, p=p),
            metric_args={"p": p},
        )

    def test_minkowski_distance_half_cpu(self, preds, target, ref_metric, p):
        """Test dtype support of the metric on CPU."""
        self.run_precision_test_cpu(preds, target, MinkowskiDistance, minkowski_distance, metric_args={"p": p})

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_minkowski_distance_half_gpu(self, preds, target, ref_metric, p):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(preds, target, MinkowskiDistance, minkowski_distance, metric_args={"p": p})


def test_error_on_different_shape():
    """Test that error is raised on different shapes of input."""
    metric = MinkowskiDistance(5.1)
    with pytest.raises(RuntimeError, match="Predictions and targets are expected to have the same shape"):
        metric(torch.randn(50), torch.randn(100))


def test_error_on_wrong_p_arg():
    """Test that error is raised if wrongly p argument is provided."""
    with pytest.raises(TorchMetricsUserError, match="Argument ``p`` must be a float.*"):
        MinkowskiDistance(p=-10)
