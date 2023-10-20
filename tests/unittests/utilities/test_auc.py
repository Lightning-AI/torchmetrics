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
from typing import NamedTuple

import numpy as np
import pytest
from sklearn.metrics import auc as _sk_auc
from torch import Tensor, tensor
from torchmetrics.utilities.compute import auc
from unittests import NUM_BATCHES
from unittests.helpers import seed_all
from unittests.helpers.testers import MetricTester

seed_all(42)


class _Input(NamedTuple):
    x: Tensor
    y: Tensor


def sk_auc(x, y, reorder=False):
    """Comparison function for correctness of auc implementation."""
    x = x.flatten()
    y = y.flatten()
    if reorder:
        idx = np.argsort(x, kind="stable")
        x = x[idx]
        y = y[idx]
    return _sk_auc(x, y)


_examples = []
# generate already ordered samples, sorted in both directions
for batch_size in (8, 4049):
    for i in range(4):
        x = np.random.rand(NUM_BATCHES * batch_size)
        y = np.random.rand(NUM_BATCHES * batch_size)
        idx = np.argsort(x, kind="stable")
        x = x[idx] if i % 2 == 0 else x[idx[::-1]]
        y = y[idx] if i % 2 == 0 else x[idx[::-1]]
        x = x.reshape(NUM_BATCHES, batch_size)
        y = y.reshape(NUM_BATCHES, batch_size)
        _examples.append(_Input(x=tensor(x), y=tensor(y)))


@pytest.mark.parametrize("x, y", _examples)
class TestAUC(MetricTester):
    """Test class for `AUC`."""

    @pytest.mark.parametrize("reorder", [True, False])
    def test_auc_functional(self, x, y, reorder):
        """Test functional implementation."""
        self.run_functional_metric_test(
            x,
            y,
            metric_functional=auc,
            reference_metric=partial(sk_auc, reorder=reorder),
            metric_args={"reorder": reorder},
        )


@pytest.mark.parametrize("unsqueeze_x", [True, False])
@pytest.mark.parametrize("unsqueeze_y", [True, False])
@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        ([0, 1], [0, 1], 0.5),
        ([1, 0], [0, 1], 0.5),
        ([1, 0, 0], [0, 1, 1], 0.5),
        ([0, 1], [1, 1], 1),
        ([0, 0.5, 1], [0, 0.5, 1], 0.5),
    ],
)
def test_auc(x, y, expected, unsqueeze_x, unsqueeze_y):
    """Test that auc function gives the expected result."""
    x = tensor(x)
    y = tensor(y)

    if unsqueeze_x:
        x = x.unsqueeze(-1)

    if unsqueeze_y:
        y = y.unsqueeze(-1)

    # Test Area Under Curve (AUC) computation
    assert auc(x, y, reorder=True) == expected
