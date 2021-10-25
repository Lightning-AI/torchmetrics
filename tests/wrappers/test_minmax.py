from functools import partial

import pytest
import torch

from tests.helpers import seed_all
from tests.helpers.testers import BATCH_SIZE, NUM_BATCHES, NUM_CLASSES, MetricTester
from torchmetrics import Accuracy, ConfusionMatrix, MeanSquaredError
from torchmetrics.wrappers import MinMaxMetric

seed_all(42)


class TestingMinMaxMetric(MinMaxMetric):
    """wrap metric to fit testing framework."""

    def compute(self):
        """instead of returning dict, return as list."""
        output_dict = super().compute()
        return [output_dict["raw"], output_dict["min"], output_dict["max"]]

    def forward(self, *args, **kwargs):
        self.update(*args, **kwargs)
        return self.compute()


def compare_fn(preds, target, base_fn):
    """comparing function for minmax wrapper."""
    min, max = 1e6, -1e6  # pick some very large numbers for comparing
    for i in range(NUM_BATCHES):
        val = base_fn(preds[: (i + 1) * BATCH_SIZE], target[: (i + 1) * BATCH_SIZE]).cpu().numpy()
        min = min if min < val else val
        max = max if max > val else val
    raw = base_fn(preds, target)
    return [raw.cpu().numpy(), min, max]


@pytest.mark.parametrize(
    "preds, target, base_metric",
    [
        (
            torch.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES).softmax(dim=-1),
            torch.randint(NUM_CLASSES, (NUM_BATCHES, BATCH_SIZE)),
            Accuracy(num_classes=NUM_CLASSES),
        ),
        (torch.randn(NUM_BATCHES, BATCH_SIZE), torch.randn(NUM_BATCHES, BATCH_SIZE), MeanSquaredError()),
    ],
)
class TestMultioutputWrapper(MetricTester):
    """Test the MinMaxMetric wrapper works as expected."""

    @pytest.mark.parametrize("ddp", [True, False])
    def test_multioutput_wrapper(self, preds, target, base_metric, ddp):
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            TestingMinMaxMetric,
            partial(compare_fn, base_fn=base_metric),
            dist_sync_on_step=False,
            metric_args=dict(base_metric=base_metric),
            check_batch=False,
            check_scriptable=False,
        )

@pytest.mark.parametrize(
    "preds, labels, raws, maxs, mins", 
    [
        (
            ([[0.9, 0.1], [0.2, 0.8]],[[0.1, 0.9], [0.2, 0.8]], [[0.1, 0.9], [0.8, 0.2]]),
            [[0, 1], [0, 1]],
            (0.5, 1.0, 0.5),
            (0.5, 1.0, 1.0),
            (0.5, 0.5, 0.5)
        )
    ])
def test_basic_example(preds, labels, raws, maxs, mins) -> None:
    """tests that both min and max versions of MinMaxMetric operate correctly after calling compute."""
    acc = Accuracy()
    min_max_acc = MinMaxMetric(acc)

    preds_1 = torch.Tensor(preds[0])
    preds_2 = torch.Tensor(preds[1])
    preds_3 = torch.Tensor(preds[2])
    labels = torch.Tensor(labels).long()

    min_max_acc(preds_1, labels)
    acc = min_max_acc.compute()
    assert acc["raw"] == raws[0]
    assert acc["max"] == maxs[0]
    assert acc["min"] == mins[0]

    min_max_acc(preds_2, labels)
    acc = min_max_acc.compute()
    assert acc["raw"] == raws[1]
    assert acc["max"] == maxs[1]
    assert acc["min"] == mins[1]

    min_max_acc(preds_3, labels)
    acc = min_max_acc.compute()
    assert acc["raw"] == raws[2]
    assert acc["max"] == maxs[2]
    assert acc["min"] == mins[2]


def test_no_base_metric() -> None:
    """tests that ValueError is raised when no base_metric is passed."""
    with pytest.raises(ValueError, match=r"Expected base metric to be an instance .*"):
        MinMaxMetric([])


def test_no_scalar_compute() -> None:
    """tests that an assertion error is thrown if the wrapped basemetric gives a non-scalar on compute."""
    min_max_nsm = MinMaxMetric(ConfusionMatrix(num_classes=2))

    with pytest.raises(RuntimeError, match=r"Returned value from base metric should be a scalar .*"):
        min_max_nsm.compute()
