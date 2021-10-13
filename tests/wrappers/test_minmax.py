import pytest
import torch

from torchmetrics.classification import Accuracy
from torchmetrics.wrappers import MinMaxMetric


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("type", ["min", "max"])
def test_minmax(device, type):
    """test that both min and max versions of MinMaxMetric operate correctly after calling compute."""
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
