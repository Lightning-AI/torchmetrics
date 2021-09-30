import pytest
import torch


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("type", ["min", "max"])
def test_minmax(device, type):
    """test that both min and max versions of MinMaxMetric operate correctly after calling compute."""
    m = MinMaxMetric()
    acc = Accuracy()

    preds_1 = torch.Tensor([[0.9, 0.1], [0.2, 0.8]])
    preds_2 = torch.Tensor([[0.1, 0.9], [0.2, 0.8]])
    preds_3 = torch.Tensor([[0.1, 0.9], [0.8, 0.2]])
    labels = torch.Tensor([[0, 1], [0, 1]]).long()

    acc(preds_1, labels)  # acc is 0.5
    m(acc.compute())  # max_metrix is 0.5
    assert m.compute() == 0.5

    acc(preds_2, labels)  # acc is 1.
    m(acc.compute())  # max_metrix is 1.
    assert m.compute() == 1.0

    acc(preds_3, labels)  # acc is 0.5
    m(acc.compute())  # max_metrix is 1.
    assert m.compute() == 1.0
