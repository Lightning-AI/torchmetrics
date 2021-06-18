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
from collections import namedtuple
from functools import partial
from typing import Callable

import pytest
import torch
from torch import Tensor


def test_import_speechmetrics() -> None:
    try:
        import speechmetrics
    except ImportError:
        pytest.fail('ImportError speechmetrics')


from tests.helpers import seed_all
from tests.helpers.testers import BATCH_SIZE, NUM_BATCHES, MetricTester

seed_all(42)

Time = 100

Input = namedtuple('Input', ["preds", "target"])

inputs = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, 1, Time),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE, 1, Time),
)

import multiprocessing
import time
import speechmetrics

speechmetrics_sisdr = speechmetrics.load('sisdr')
def speechmetrics_si_sdr(preds: Tensor, target: Tensor,
                            zero_mean: bool) -> Tensor:
    if zero_mean:
        preds = preds - preds.mean(dim=2, keepdim=True)
        target = target - target.mean(dim=2, keepdim=True)
    target = target.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()
    mss = []
    for i in range(preds.shape[0]):
        ms = []
        for j in range(preds.shape[1]):
            metric = speechmetrics_sisdr(preds[i, j],
                                            target[i, j],
                                            rate=16000)
            ms.append(metric['sisdr'][0])
        mss.append(ms)
    return torch.tensor(mss)

def test_speechmetrics_si_sdr() -> None:
    t = multiprocessing.Process(target=speechmetrics_si_sdr,
                                args=(inputs.preds[0], inputs.target[0],
                                      False))
    t.start()
    try:
        t.join(timeout=180)  # 3min
        if t.is_alive():
            pytest.fail(f'timeout 3min. t.is_alive()={t.is_alive()}')
            t.terminate()
    except:
        pytest.fail('join except')
