from torch import tensor

from torchmetrics import RetrievalRecallAtFixedPrecision


class TestRetrievalRecallAtFixedPrecision:
    def test_test(self):
        indexes = tensor([0, 0, 0, 0, 1, 1, 1])
        preds = tensor([0.4, 0.01, 0.5, 0.6, 0.2, 0.3, 0.5])
        target = tensor([True, False, False, True, True, False, True])
        r = RetrievalRecallAtFixedPrecision(min_precision=0.8)

        assert (tensor(0.5), 1) == r(preds, target, indexes=indexes)
