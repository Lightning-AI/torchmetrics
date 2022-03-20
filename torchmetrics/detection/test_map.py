import unittest

import torch

from torchmetrics.detection.map import MeanAveragePrecision


class TestMap(unittest.TestCase):
    @staticmethod
    def _compute_metric(gts, preds):
        metric = MeanAveragePrecision()
        metric.reset()
        metric.update(preds, gts)
        return metric.compute()

    def test_basic_correct_detection(self):
        """Basic test, 1 ground truth, 1 good prediction."""
        gt = dict(boxes=torch.Tensor([[10, 20, 15, 25]]), labels=torch.IntTensor([0]))
        pred = dict(
                boxes=torch.Tensor([[10, 20, 15, 25]]),
                scores=torch.Tensor([0.9]),
                labels=torch.IntTensor([0]),
            )

        result = self._compute_metric([gt], [pred])
        self.assertEqual(result["map"], 1)

    def test_no_gt_no_pred(self):
        """Image should be ignored, so map result is -1 (the default value)"""
        gt = dict(
                boxes=torch.Tensor([]),  # Empty GT
                labels=torch.IntTensor([]),
            )

        pred = dict(
                boxes=torch.Tensor([]),
                scores=torch.Tensor([]),
                labels=torch.IntTensor([]),  # Empty prediction
            )

        result = self._compute_metric([gt], [pred])
        self.assertEqual(result["map"], -1)

    def test_missing_pred(self):
        """One good detection, one false negative.

        Map should be lower than 1. Actually it is 0.5, but the exact value depends on where we are sampling (i.e. recall's
        values)
        """
        gts = [
            dict(boxes=torch.Tensor([[10, 20, 15, 25]]), labels=torch.IntTensor([0])),
            dict(boxes=torch.Tensor([[10, 20, 15, 25]]), labels=torch.IntTensor([0])),
        ]
        preds = [
            dict(
                boxes=torch.Tensor([[10, 20, 15, 25]]),
                scores=torch.Tensor([0.9]),
                labels=torch.IntTensor([0]),
            ),
            dict(
                boxes=torch.Tensor([]),
                scores=torch.Tensor([]),
                labels=torch.IntTensor([]),  # Empty prediction
            ),
        ]

        result = self._compute_metric(gts, preds)
        self.assertLess(result["map"], 1)

    def test_missing_gt(self):
        """The symmetric case of test_missing_pred.

        One good detection, one false positive. Map should be lower than 1. Actually it is 0.5, but the exact value depends on
        where we are sampling (i.e. recall's values)
        """
        gts = [
            dict(boxes=torch.Tensor([[10, 20, 15, 25]]), labels=torch.IntTensor([0])),
            dict(
                boxes=torch.Tensor([]),
                labels=torch.IntTensor([]),
            ),
        ]

        preds = [
            dict(
                boxes=torch.Tensor([[10, 20, 15, 25]]),
                scores=torch.Tensor([0.9]),
                labels=torch.IntTensor([0]),
            ),
            dict(
                boxes=torch.Tensor([[10, 20, 15, 25]]),
                scores=torch.Tensor([0.95]),
                labels=torch.IntTensor([0]),
            ),
        ]

        result = self._compute_metric(gts, preds)
        self.assertLess(result["map"], 1)


if __name__ == "__main__":
    unittest.main()
