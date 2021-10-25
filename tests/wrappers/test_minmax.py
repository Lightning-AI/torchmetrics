import pytest
import torch

from torchmetrics.classification import Accuracy, ConfusionMatrix
from torchmetrics.metric import Metric
from torchmetrics.wrappers import MinMaxMetric


def test_base() -> None:
    """tests that both min and max versions of MinMaxMetric operate correctly after calling compute."""
    acc = Accuracy()
    min_max_acc = MinMaxMetric(acc)

    preds_1 = torch.Tensor([[0.9, 0.1], [0.2, 0.8]])
    preds_2 = torch.Tensor([[0.1, 0.9], [0.2, 0.8]])
    preds_3 = torch.Tensor([[0.1, 0.9], [0.8, 0.2]])
    labels = torch.Tensor([[0, 1], [0, 1]]).long()

    min_max_acc(preds_1, labels)
    acc = min_max_acc.compute()
    assert acc["raw"] == 0.5
    assert acc["max"] == 0.5
    assert acc["min"] == 0.5

    min_max_acc(preds_2, labels)
    acc = min_max_acc.compute()
    assert acc["raw"] == 1.0
    assert acc["max"] == 1.0
    assert acc["min"] == 0.5

    min_max_acc(preds_3, labels)
    acc = min_max_acc.compute()
    assert acc["raw"] == 0.5
    assert acc["max"] == 1.0
    assert acc["min"] == 0.5


def test_no_base_metric() -> None:
    """tests that ValueError is raised when no base_metric is passed."""
    x = ""
    with pytest.raises(ValueError):
        MinMaxMetric(x)


def test_no_scalar_compute() -> None:
    """tests that an assertion error is thrown if the wrapped basemetric gives a non-scalar on compute."""

    nsm = ConfusionMatrix(num_classes=2)
    min_max_nsm = MinMaxMetric(nsm)

    with pytest.raises(AssertionError):
        min_max_nsm.compute()
