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

import numpy as np
import pytest
import torch
from sewar.full_ref import vifp
from torchmetrics.functional.image.vif import visual_information_fidelity
from torchmetrics.image.vif import VisualInformationFidelity

from unittests import BATCH_SIZE, NUM_BATCHES, _Input
from unittests.helpers import seed_all
from unittests.helpers.testers import MetricTester

seed_all(42)


_inputs = [
    _Input(
        preds=torch.randint(0, 255, size=(NUM_BATCHES, BATCH_SIZE, channels, 41, 41), dtype=torch.float),
        target=torch.randint(0, 255, size=(NUM_BATCHES, BATCH_SIZE, channels, 41, 41), dtype=torch.float),
    )
    for channels in [1, 3]
]


def _sewar_vif(preds, target, sigma_nsq=2):
    preds = torch.movedim(preds, 1, -1)
    target = torch.movedim(target, 1, -1)
    preds = preds.cpu().numpy()
    target = target.cpu().numpy()
    vif = [vifp(GT=target[batch], P=preds[batch], sigma_nsq=sigma_nsq) for batch in range(preds.shape[0])]
    return np.mean(vif)


@pytest.mark.parametrize("preds, target", [(inputs.preds, inputs.target) for inputs in _inputs])
class TestVIF(MetricTester):
    """Test class for `VisualInformationFidelity` metric."""

    atol = 1e-7

    @pytest.mark.parametrize("ddp", [True, False])
    def test_vif(self, preds, target, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(ddp, preds, target, VisualInformationFidelity, _sewar_vif)

    def test_vif_functional(self, preds, target):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(preds, target, visual_information_fidelity, _sewar_vif)
