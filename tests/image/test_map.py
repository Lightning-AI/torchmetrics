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

import torch

from torchmetrics.image.map import MAP


class TestMapMetric(unittest.TestCase):
    def generate_predictions_targets(self, batch_size):
        box1 = [0.0, 0.0, 1.0, 1.0]  # TP_class0
        box2 = [1.0, 1.0, 2.0, 2.0]  # TP_class0
        box3 = [2.0, 2.0, 3.0, 3.0]  # false class (FN for class_0, FP for class_1?)
        box4 = [3.0, 3.0, 4.0, 4.0]  # FN (missing box for class_0)
        box5 = [4.0, 4.0, 5.0, 5.0]  # FP (detection - but no GT for class_0)
        box6 = [5.0, 5.0, 6.0, 6.0]  # TP_class_2 --> to check if we get precision one for class_2
        scores = [0.8, 0.9, 0.7, 1.0, 0.6, 1.0]

        targets = [{
            "groundtruth_boxes": torch.tensor([[box1[0:4], box2[0:4], box3[0:4], box4[0:4], box6[0:4]]]),
            "groundtruth_classes": torch.tensor([[0, 0, 1, 0, 2]]),
        }] * batch_size
        predictions = [{
            "detection_boxes": torch.tensor([[box1, box2, box3, box5, box6]]),
            "detection_classes": torch.tensor([[0, 0, 0, 0, 2]]),
            "detection_scores": torch.tensor([[scores]]),
        }] * batch_size
        # How to calculate expected_values:
        # sorted by score both classes:
        # box 2 TP
        # box 1 TP
        # box 3 FN
        # box 5 FN

        # sorted by score class_0:
        # box 2 TP
        # box 1 TP
        # box 5 FN

        # sorted by score class_1:
        # box 3  missing

        # sorted by score class_2:
        # box 6 ok --> 1 gt/1 prediction --> 100%

        return predictions, targets

    def test_mAP_average(self):
        numerical_correction = 201 / 202  # (1-100/101)/2 --> 101 coming from interpolation)
        expected_value = 1 / 3 * 1 + 1 / 3 * 0 + 1 / 3 * 2 / 3 * numerical_correction

        expected_value_class0 = (2 / 3) * numerical_correction
        expected_value_class1 = 0 / 1
        expected_value_class2 = 1 / 1

        for batch_size in [2, 10]:
            predictions, targets = self.generate_predictions_targets(batch_size=batch_size)

            mAP = MAP(num_classes=3)
            mAP.update(predictions, targets)
            mAP.update(predictions, targets)
            mAP.update(predictions, targets)  # dist_reduce_fx='mean' -> should stay the same value
            metrics = mAP.compute()
            map_value = metrics.map_value
            map_values_class = metrics.map_per_class_value

            # test if reduction in the Metric is set to mean
            self.assertAlmostEqual(map_value.item(), expected_value, delta=1e-05)
            self.assertAlmostEqual(map_values_class[0].item(), expected_value_class0, delta=1e-05)
            self.assertAlmostEqual(map_values_class[1].item(), expected_value_class1, delta=1e-05)
            self.assertAlmostEqual(map_values_class[2].item(), expected_value_class2, delta=1e-05)
