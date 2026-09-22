# Copyright The Lightning team.
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


# from sklearn.metrics import brier as sk_brier
from torch import tensor

from torchmetrics.classification.brier import (
    BinaryBrier,
    MulticlassBrier,
)

# from torchmetrics.functional.classification.brier import (
#    binary_brier,
#    multiclass_brier,
#    multilabel_brier,
# )
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

seed_all(42)


class TestBinaryBrier(MetricTester):
    """Test class for `BinaryBrier` metric."""

    def test_binary_brier(self):
        """Test class implementation of metric."""
        target = tensor([0, 1, 0, 1, 0, 1])
        preds = tensor([0.11, 0.22, 0.84, 0.73, 0.33, 0.92])
        metric = BinaryBrier()
        print("\n")
        print("BinaryBrier:")
        print(metric(preds, target))
        print("\n")


class TestMulticlassBrier(MetricTester):
    """Test class for `BinaryBrier` metric."""

    def test_multiclass_brier(self):
        """Test class implementation of metric."""
        target = tensor([1, 4, 3, 0, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0])
        preds = tensor([
            [0.9, 0.05, 0.02, 0.01, 0.02],
            [0.01, 0.01, 0.01, 0.01, 0.96],
            [0.85, 0.05, 0.05, 0.02, 0.03],
            [0.01, 0.01, 0.9, 0.05, 0.03],
            [0.8, 0.1, 0.05, 0.03, 0.02],
            [0.01, 0.01, 0.01, 0.9, 0.07],
            [0.05, 0.1, 0.8, 0.03, 0.02],
            [0.02, 0.02, 0.02, 0.9, 0.04],
            [0.01, 0.01, 0.01, 0.02, 0.95],
            [0.7, 0.2, 0.05, 0.03, 0.02],
            [0.02, 0.9, 0.05, 0.02, 0.01],
            [0.1, 0.2, 0.6, 0.05, 0.05],
            [0.03, 0.03, 0.03, 0.9, 0.01],
            [0.02, 0.02, 0.02, 0.02, 0.92],
            [0.75, 0.15, 0.05, 0.03, 0.02],
            [0.02, 0.85, 0.1, 0.02, 0.01],
            [0.1, 0.15, 0.7, 0.03, 0.02],
            [0.02, 0.02, 0.02, 0.92, 0.02],
            [0.01, 0.01, 0.01, 0.03, 0.94],
            [0.8, 0.1, 0.05, 0.03, 0.02],
        ])
        metric = MulticlassBrier(num_classes=5)
        print("MulticlassBrier:")
        print(metric(preds, target))
        print("\n")
