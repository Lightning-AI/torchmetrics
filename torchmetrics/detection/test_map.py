# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest

from torch import IntTensor, Tensor

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
        gt = dict(boxes=Tensor([[10, 20, 15, 25]]), labels=IntTensor([0]))
        pred = dict(boxes=Tensor([[10, 20, 15, 25]]), scores=Tensor([0.9]), labels=IntTensor([0]))

        result = self._compute_metric([gt], [pred])
        self.assertEqual(result["map"], 1)

    def test_no_gt_no_pred(self):
        """Image should be ignored, so map result is -1 (the default value)"""
        # Empty GT
        gt = dict(boxes=Tensor([]), labels=IntTensor([]))
        # Empty prediction
        pred = dict(boxes=Tensor([]), scores=Tensor([]), labels=IntTensor([]))

        result = self._compute_metric([gt], [pred])
        self.assertEqual(result["map"], -1)

    def test_missing_pred(self):
        """One good detection, one false negative.

        Map should be lower than 1. Actually it is 0.5, but the exact value depends on where we are sampling (i.e.
        recall's values)
        """
        gts = [
            dict(boxes=Tensor([[10, 20, 15, 25]]), labels=IntTensor([0])),
            dict(boxes=Tensor([[10, 20, 15, 25]]), labels=IntTensor([0])),
        ]
        preds = [
            dict(boxes=Tensor([[10, 20, 15, 25]]), scores=Tensor([0.9]), labels=IntTensor([0])),
            # Empty prediction
            dict(boxes=Tensor([]), scores=Tensor([]), labels=IntTensor([])),
        ]

        result = self._compute_metric(gts, preds)
        self.assertLess(result["map"], 1)

    def test_missing_gt(self):
        """The symmetric case of test_missing_pred.

        One good detection, one false positive. Map should be lower than 1. Actually it is 0.5, but the exact value
        depends on where we are sampling (i.e. recall's values)
        """
        gts = [
            dict(boxes=Tensor([[10, 20, 15, 25]]), labels=IntTensor([0])),
            dict(boxes=Tensor([]), labels=IntTensor([])),
        ]
        preds = [
            dict(boxes=Tensor([[10, 20, 15, 25]]), scores=Tensor([0.9]), labels=IntTensor([0])),
            dict(boxes=Tensor([[10, 20, 15, 25]]), scores=Tensor([0.95]), labels=IntTensor([0])),
        ]

        result = self._compute_metric(gts, preds)
        self.assertLess(result["map"], 1)
