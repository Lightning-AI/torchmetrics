import numpy as np
import pytest
import torch

from tests.helpers.testers import BATCH_SIZE, NUM_BATCHES, MetricTester
from torchmetrics.average import AverageMeter


def average(values, weights):
    return np.average(values, weights=weights)


def average_ignore_weights(values, weights):
    return np.average(values)


class DefaultWeightWrapper(AverageMeter):
    def update(self, values, weights):
        super().update(values)


class ScalarWrapper(AverageMeter):
    def update(self, values, weights):
        # torch.ravel is PyTorch 1.8 only, so use np.ravel instead
        values = values.cpu().numpy()
        weights = weights.cpu().numpy()
        for v, w in zip(np.ravel(values), np.ravel(weights)):
            super().update(float(v), float(w))


@pytest.mark.parametrize(
    "values, weights",
    [
        (torch.rand(NUM_BATCHES, BATCH_SIZE), torch.ones(NUM_BATCHES, BATCH_SIZE)),
        (torch.rand(NUM_BATCHES, BATCH_SIZE), torch.rand(NUM_BATCHES, BATCH_SIZE) > 0.5),
        (torch.rand(NUM_BATCHES, BATCH_SIZE, 2), torch.rand(NUM_BATCHES, BATCH_SIZE, 2) > 0.5),
    ],
)
class TestAverageMeter(MetricTester):
    @pytest.mark.parametrize("ddp", [False, True])
    @pytest.mark.parametrize("dist_sync_on_step", [False, True])
    def test_average_fn(self, ddp, dist_sync_on_step, values, weights):
        self.run_class_metric_test(
            ddp=ddp,
            dist_sync_on_step=dist_sync_on_step,
            metric_class=AverageMeter,
            sk_metric=average,
            # Abuse of names here
            preds=values,
            target=weights,
        )

    @pytest.mark.parametrize("ddp", [False, True])
    @pytest.mark.parametrize("dist_sync_on_step", [False, True])
    def test_average_fn_default(self, ddp, dist_sync_on_step, values, weights):
        self.run_class_metric_test(
            ddp=ddp,
            dist_sync_on_step=dist_sync_on_step,
            metric_class=DefaultWeightWrapper,
            sk_metric=average_ignore_weights,
            # Abuse of names here
            preds=values,
            target=weights,
        )

    @pytest.mark.parametrize("ddp", [False, True])
    @pytest.mark.parametrize("dist_sync_on_step", [False, True])
    def test_average_fn_scalar(self, ddp, dist_sync_on_step, values, weights):
        self.run_class_metric_test(
            ddp=ddp,
            dist_sync_on_step=dist_sync_on_step,
            metric_class=ScalarWrapper,
            sk_metric=average,
            # Abuse of names here
            preds=values,
            target=weights,
        )


@pytest.mark.skipif(not hasattr(torch, "broadcast_to"), reason="PyTorch <1.8 does not have broadcast_to")
@pytest.mark.parametrize(
    "weights, expected", [(1, 11.5), (torch.ones(2, 1, 1), 11.5), (torch.tensor([1, 2]).reshape(2, 1, 1), 13.5)]
)
def test_AverageMeter_broadcasting(weights, expected):
    values = torch.arange(24).reshape(2, 3, 4)
    avg = AverageMeter()

    assert avg(values, weights) == expected
