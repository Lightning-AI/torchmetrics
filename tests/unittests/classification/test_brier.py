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
from functools import partial

import numpy as np
import pytest
import torch
from scipy.special import expit as sigmoid

# from sklearn.metrics import brier as sk_brier
from torch import tensor

from torchmetrics.classification.brier import (
    BinaryBrier,
    Brier,
    MulticlassBrier,
    MultilabelBrier,
)

# from torchmetrics.functional.classification.brier import (
#    binary_brier,
#    multiclass_brier,
#    multilabel_brier,
# )
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_2_1
from unittests import NUM_CLASSES, THRESHOLD
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester, inject_ignore_index, remove_ignore_index
from unittests.classification._inputs import _binary_cases, _multiclass_cases, _multilabel_cases

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
        target = tensor([1, 5, 3, 0, 0, 3])
        preds = tensor([
            [0.9, 0.05, 0.02, 0.01, 0.01, 0.01],
            [0.01, 0.01, 0.01, 0.01, 0.01, 0.95],
            [0.85, 0.05, 0.05, 0.02, 0.02, 0.01],
            [0.01, 0.01, 0.9, 0.05, 0.02, 0.01],
            [0.8, 0.1, 0.05, 0.03, 0.01, 0.01],
            [0.01, 0.01, 0.01, 0.9, 0.05, 0.02],
        ])
        metric = MulticlassBrier(num_classes=6)
        print("MulticlassBrier:")
        print(metric(preds, target))
        print("\n")
