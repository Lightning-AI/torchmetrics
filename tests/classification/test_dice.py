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

import pytest
import torch
from torch import tensor, Tensor

from tests.classification.inputs import _input_multiclass_prob as _input_mcls_prob
from tests.helpers.testers import MetricTester
from tests.helpers import seed_all
from torchmetrics import Dice
from torchmetrics.functional.classification.dice import _stat_scores
from torchmetrics.functional import dice_score
from torchmetrics.functional import dice


seed_all(42)


def _dice_score(
    preds: Tensor,
    target: Tensor,
    background: bool = False,
    nan_score: float = 0.0,
) -> Tensor:
    """
    Compute dice score from prediction scores.
    There is no implementation of Dice in sklearn. I used public information about
    metric: `https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient`

    Args:
        preds: prediction tensor
        target: target tensor
        background: whether to also compute dice for the background
        nan_score: the value to use for the score if denominator equals zero

    Return:
        Tensor containing dice score

    """
    num_classes = preds.shape[1]
    bg_inv = 1 - int(background)

    tp = tensor(0, device=preds.device)
    fp = tensor(0, device=preds.device)
    fn = tensor(0, device=preds.device)

    for i in range(bg_inv, num_classes):
        tp_cls, fp_cls, _, fn_cls, _ = _stat_scores(preds=preds, target=target, class_index=i)

        tp += tp_cls
        fp += fp_cls
        fn += fn_cls

    denom = (2 * tp + fp + fn).to(torch.float)
    score = (2 * tp).to(torch.float) / denom if torch.is_nonzero(denom) else nan_score
    return score


@pytest.mark.parametrize(
    ["pred", "target", "expected"],
    [
        ([[0, 0], [1, 1]], [[0, 0], [1, 1]], 1.0),
        ([[1, 1], [0, 0]], [[0, 0], [1, 1]], 0.0),
        ([[1, 1], [1, 1]], [[1, 1], [0, 0]], 2 / 3),
        ([[1, 1], [0, 0]], [[1, 1], [0, 0]], 1.0),
    ],
)
def test_dice_score(pred, target, expected):
    score = dice_score(tensor(pred), tensor(target))
    assert score == expected


@pytest.mark.parametrize(
    ["pred", "target", "expected"],
    [
        ([[0, 0], [1, 1]], [[0, 0], [1, 1]], 1.0),
        ([[1, 1], [0, 0]], [[0, 0], [1, 1]], 0.0),
        ([[1, 1], [1, 1]], [[1, 1], [0, 0]], 2 / 3),
        ([[1, 1], [0, 0]], [[1, 1], [0, 0]], 1.0),
    ],
)
def test_dice(pred, target, expected):
    score = dice(tensor(pred), tensor(target))
    assert score == expected


@pytest.mark.parametrize(
    "preds, target",
    [
        (_input_mcls_prob.preds, _input_mcls_prob.target)
    ],
)
@pytest.mark.parametrize("background", [False, True])
class TestDice(MetricTester):
    @pytest.mark.parametrize("ddp", [False])
    @pytest.mark.parametrize("dist_sync_on_step", [False])
    def test_dice_class(self, ddp, dist_sync_on_step, preds, target, background):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=Dice,
            sk_metric=partial(_dice_score, background=background),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={"background": background},
        )

    def test_dice_fn(self, preds, target, background):
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=dice,
            sk_metric=partial(_dice_score, background=background),
            metric_args={"background": background},
        )
