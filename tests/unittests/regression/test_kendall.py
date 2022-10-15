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

import pytest
import torch
from scipy.stats import kendalltau

from torchmetrics.functional.regression.kendall import kendall_rank_corrcoef
from torchmetrics.regression.kendall import KendallRankCorrCoef
from unittests.helpers import seed_all
from unittests.helpers.testers import BATCH_SIZE, EXTRA_DIM, NUM_BATCHES, MetricTester

seed_all(42)

Input = namedtuple("Input", ["preds", "target"])
_single_inputs1 = Input(preds=torch.rand(NUM_BATCHES, BATCH_SIZE), target=torch.rand(NUM_BATCHES, BATCH_SIZE))
_single_inputs2 = Input(preds=torch.randn(NUM_BATCHES, BATCH_SIZE), target=torch.randn(NUM_BATCHES, BATCH_SIZE))
_single_inputs3 = Input(
    preds=torch.randint(-10, 10, (NUM_BATCHES, BATCH_SIZE)), target=torch.randint(-10, 10, (NUM_BATCHES, BATCH_SIZE))
)
_multi_inputs1 = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM), target=torch.rand(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)
)
_multi_inputs2 = Input(
    preds=torch.randn(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM), target=torch.randn(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)
)
_multi_inputs3 = Input(
    preds=torch.randint(-10, 10, (NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)),
    target=torch.randint(-10, 10, (NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)),
)


def _sk_metric(preds, target, alternative="two-sided"):
    _alternative = alternative or "two-sided"
    if preds.ndim == 2:
        out = [
            kendalltau(p.numpy(), t.numpy(), method="asymptotic", alternative=_alternative)
            for p, t in zip(preds, target)
        ]
        tau = torch.cat([torch.tensor(o[0]).unsqueeze(0) for o in out])
        p_value = torch.cat([torch.tensor(o[1]).unsqueeze(0) for o in out])
        if alternative is not None:
            return tau, p_value
        return tau

    tau, p_value = kendalltau(preds.numpy(), target.numpy(), method="asymptotic", alternative=_alternative)

    if alternative is not None:
        return torch.tensor(tau), torch.tensor(p_value)
    return torch.tensor(tau)


@pytest.mark.parametrize(
    "preds, target, alternative",
    [
        (_single_inputs1.preds, _single_inputs1.target, None),
        (_single_inputs2.preds, _single_inputs2.target, "less"),
        (_single_inputs3.preds, _single_inputs3.target, "greater"),
        (_multi_inputs1.preds, _multi_inputs1.target, None),
        (_multi_inputs2.preds, _multi_inputs2.target, "two-sided"),
        (_multi_inputs3.preds, _multi_inputs3.target, "greater"),
    ],
)
class TestKendallRankCorrCoef(MetricTester):
    @pytest.mark.parametrize("ddp", [False])
    @pytest.mark.parametrize("dist_sync_on_step", [False])
    def test_kendall_rank_corrcoef(self, preds, target, alternative, ddp, dist_sync_on_step):
        num_outputs = EXTRA_DIM if preds.ndim == 3 else 1
        t_test = True if alternative is not None else False
        _sk_kendall_tau = partial(_sk_metric, alternative=alternative)

        self.run_class_metric_test(
            ddp,
            preds,
            target,
            KendallRankCorrCoef,
            _sk_kendall_tau,
            dist_sync_on_step,
            metric_args={"t_test": t_test, "alternative": alternative, "num_outputs": num_outputs},
        )

    def test_kendall_rank_corrcoef_functional(self, preds, target, alternative):
        t_test = True if alternative is not None else False
        metric_args = {"t_test": t_test, "alternative": alternative}
        _sk_kendall_tau = partial(_sk_metric, alternative=alternative)
        self.run_functional_metric_test(preds, target, kendall_rank_corrcoef, _sk_kendall_tau, metric_args=metric_args)
