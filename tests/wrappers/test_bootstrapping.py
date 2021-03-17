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
import pytest
from functools import partial

import numpy as np
import torch

from tests.helpers.testers import MetricTester

from sklearn.metrics import precision_score, recall_score

from torchmetrics.classification import Precision, Recall
from torchmetrics.wrappers.bootstrapping import BootStrapper

_ = torch.manual_seed(0)

_preds = torch.randint(10, (10, 32))
_target = torch.randint(10, (10, 32))

def _sk_bootstrap(preds, target, func, num_bootstraps=10):
    preds = preds.numpy()
    target = target.numpy()
    
    scores = [ ]
    for i in range(num_bootstraps):
        idx = torch.multinomial(torch.ones(preds.shape[0]), num_samples=preds.shape[0], replacement=True)
        print('numpy', idx)
        preds_idx = preds[idx]
        target_idx = target[idx]
        scores.append(func(target_idx, preds_idx, average='micro'))
    scores = np.stack(scores)
    return [scores.mean(), scores.std()]

@pytest.mark.parametrize("metric, sk_metric", [
    [Precision(), precision_score],
    [Recall(), recall_score],
])
class TestBootStrapper(MetricTester):

    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_bootstrapper(self, metric, sk_metric, ddp, dist_sync_on_step):
        self.run_class_metric_test(
            ddp,
            _preds,
            _target,
            metric_class=partial(BootStrapper, base_metric=metric),
            sk_metric=partial(_sk_bootstrap, func=sk_metric),
            dist_sync_on_step=dist_sync_on_step,
        )

    