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

from functools import partial
from typing import List

import pytest
from torch import Tensor, tensor

from tests.text.helpers import INPUT_ORDER, TextTester
from torchmetrics.functional.text.meteor import meteor_score
from torchmetrics.text.meteor import METEORScore
from torchmetrics.utilities.imports import _NLTK_AVAILABLE

if _NLTK_AVAILABLE:
    from nltk.translate.meteor_score import meteor_score as nltk_meteor_score


# Examples taken from https://github.com/nltk/nltk/blob/develop/nltk/translate/meteor_score.py
HYPOTHESIS_A = "It is a guide to action which ensures that the military always obeys the commands of the party"
HYPOTHESIS_B = "It is to insure the troops forever hearing the activity guidebook that party direct"
REFERENCE_1 = "It is a guide to action that ensures that the military will forever heed Party commands"
REFERENCE_2 = (
    "It is the guiding principle which guarantees the military forces always being under the command of the Party"
)
REFERENCE_3 = "It is the practical guide for the army always to heed the directions of the party"

REFERENCES = [[[REFERENCE_1, REFERENCE_2, REFERENCE_3]], [[REFERENCE_1, REFERENCE_2, REFERENCE_3]]]
HYPOTHESES = [[HYPOTHESIS_A], [HYPOTHESIS_B]]


def _compute_nltk_meteor_score(
    targets: List[List[str]], preds: List[str], alpha: float, beta: float, gamma: float
) -> Tensor:
    meteor_score: List[Tensor] = []
    for target, pred in zip(targets, preds):
        pred = pred.split()
        target = [t.split() for t in target]
        original_score = nltk_meteor_score(target, pred, alpha=alpha, beta=beta, gamma=gamma)
        meteor_score.append(tensor(round(original_score, 4)))
    return tensor(meteor_score)


@pytest.mark.skipif(not _NLTK_AVAILABLE, reason="test requires nltk")
@pytest.mark.parametrize(
    ["alpha", "beta", "gamma"], [pytest.param(1.0, 1.0, 1.0), pytest.param(0.9, 3.0, 0.5), pytest.param(0.5, 5.0, 0.2)]
)
@pytest.mark.parametrize(
    ["preds", "targets"],
    [
        pytest.param(HYPOTHESES, REFERENCES),
    ],
)
class TestMETEORScore(TextTester):
    atol = 1e-4

    @pytest.mark.parametrize("ddp", [False, True])
    @pytest.mark.parametrize("dist_sync_on_step", [False, True])
    def test_meteor_score_class(self, ddp, dist_sync_on_step, targets, preds, alpha, beta, gamma):
        metric_args = {"alpha": alpha, "beta": beta, "gamma": gamma}
        nltk_metric = partial(_compute_nltk_meteor_score, **metric_args)

        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            targets=targets,
            metric_class=METEORScore,
            sk_metric=nltk_metric,
            dist_sync_on_step=dist_sync_on_step,
            metric_args=metric_args,
            input_order=INPUT_ORDER.TARGETS_FIRST,
        )

    def test_meteor_score_functional(self, preds, targets, alpha, beta, gamma):
        metric_args = {"alpha": alpha, "beta": beta, "gamma": gamma}
        nltk_metric = partial(_compute_nltk_meteor_score, **metric_args)

        self.run_functional_metric_test(
            preds=preds,
            targets=targets,
            metric_functional=meteor_score,
            sk_metric=nltk_metric,
            metric_args=metric_args,
            input_order=INPUT_ORDER.PREDS_FIRST,
        )
