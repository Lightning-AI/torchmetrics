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

import numpy as np
import pytest
import torch
from sewar.full_ref import scc as sewar_scc
from torchmetrics.functional.image import spatial_correlation_coefficient
from torchmetrics.image import SpatialCorrelationCoefficient

from unittests import BATCH_SIZE, NUM_BATCHES, _Input
from unittests.helpers import seed_all
from unittests.helpers.testers import MetricTester

seed_all(42)

_inputs = [
    _Input(
        preds=torch.randn(NUM_BATCHES, BATCH_SIZE, channels, 128, 128),
        target=torch.randn(NUM_BATCHES, BATCH_SIZE, channels, 128, 128),
    )
    for channels in [1, 3]
]


def _reference_scc(preds, target):
    """Reference implementation of scc from sewar."""
    preds = torch.movedim(preds, 1, -1)
    target = torch.movedim(target, 1, -1)
    preds = preds.cpu().numpy()
    target = target.cpu().numpy()
    hp_filter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    window_size = 8
    scc = [
        sewar_scc(GT=target[batch], P=preds[batch], win=hp_filter, ws=window_size) for batch in range(preds.shape[0])
    ]
    return np.mean(scc)


@pytest.mark.parametrize("preds, target", [(i.preds, i.target) for i in _inputs])
class TestSpatialCorrelationCoefficient(MetricTester):
    """Tests for SpatialCorrelationCoefficient metric."""

    atol = 1e-3

    @pytest.mark.parametrize("ddp", [True, False])
    def test_scc(self, preds, target, ddp):
        """Test SpatialCorrelationCoefficient class usage."""
        self.run_class_metric_test(
            ddp, preds, target, metric_class=SpatialCorrelationCoefficient, reference_metric=_reference_scc
        )

    def test_scc_functional(self, preds, target):
        """Test SpatialCorrelationCoefficient functional usage."""
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=spatial_correlation_coefficient,
            reference_metric=_reference_scc,
        )
