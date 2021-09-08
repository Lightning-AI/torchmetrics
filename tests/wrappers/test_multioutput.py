import torch

from torchmetrics.wrappers.multioutput import MultioutputWrapper
from torchmetrics.classification import Accuracy
from torchmetrics.regression import R2Score


def test_multioutput_wrapper():
    # Multiple outputs, same shapes
    preds1 = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float)
    target1 = torch.tensor([[1, 4], [3, 2], [5, 6]], dtype=torch.float)
    preds2 = torch.tensor([[7, 8], [9, 10], [11, 12]], dtype=torch.float)
    target2 = torch.tensor([[7, 8], [9, 10], [11, 12]], dtype=torch.float)

    r2 = MultioutputWrapper(R2Score(), num_outputs=2)
    r2.update(preds1, target1)
    r2.update(preds2, target2)

    # R2 score computed using sklearn's r2_score
    torch.testing.assert_allclose(r2.compute(), [1, 0.8857])

    # Multiple outputs, different shapes
    acc = MultioutputWrapper(Accuracy(num_classes=3), num_outputs=2)
    preds = torch.tensor([[[0.1, 0.3], [0.8, 0.3], [0.1, 0.4]], [[0.8, 0.3], [0.1, 0.4], [0.1, 0.3]]])
    target = torch.tensor([[1, 2], [1, 1]])
    acc.update(preds, target)
    torch.testing.assert_allclose(acc.compute(), [0.5, 1.0])
