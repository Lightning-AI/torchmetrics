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
from typing import Optional

import pytest
from scipy.spatial.distance import dice as _sc_dice
from torch import Tensor, tensor

from tests.classification.inputs import _input_binary, _input_binary_logits, _input_binary_prob
from tests.classification.inputs import _input_multiclass as _input_mcls
from tests.classification.inputs import _input_multiclass_logits as _input_mcls_logits
from tests.classification.inputs import _input_multiclass_prob as _input_mcls_prob
from tests.classification.inputs import _input_multiclass_with_missing_class as _input_miss_class
from tests.classification.inputs import _input_multilabel as _input_mlb
from tests.classification.inputs import _input_multilabel_logits as _input_mlb_logits
from tests.classification.inputs import _input_multilabel_multidim as _input_mlmd
from tests.classification.inputs import _input_multilabel_multidim_prob as _input_mlmd_prob
from tests.classification.inputs import _input_multilabel_prob as _input_mlb_prob
from tests.helpers import seed_all
from tests.helpers.testers import MetricTester
from torchmetrics import Dice
from torchmetrics.functional import dice, dice_score
from torchmetrics.functional.classification.stat_scores import _del_column
from torchmetrics.utilities.checks import _input_format_classification
from torchmetrics.utilities.enums import DataType

seed_all(42)


def _sk_dice(
    preds: Tensor,
    target: Tensor,
    ignore_index: Optional[int] = None,
) -> float:
    """Compute dice score from prediction and target. Used scipy implementation of main dice logic.

    Args:
        preds: prediction tensor
        target: target tensor
        ignore_index:
            Integer specifying a target class to ignore. Recommend set to index of background class.
    Return:
        Float dice score
    """
    sk_preds, sk_target, mode = _input_format_classification(preds, target)

    if ignore_index is not None and mode != DataType.BINARY:
        sk_preds = _del_column(sk_preds, ignore_index)
        sk_target = _del_column(sk_target, ignore_index)

    sk_preds, sk_target = sk_preds.numpy(), sk_target.numpy()

    return 1 - _sc_dice(sk_preds.reshape(-1), sk_target.reshape(-1))


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
    score = dice(tensor(pred), tensor(target), ignore_index=0)
    assert score == expected


@pytest.mark.parametrize(
    "preds, target",
    [
        (_input_binary.preds, _input_binary.target),
        (_input_binary_logits.preds, _input_binary_logits.target),
        (_input_binary_prob.preds, _input_binary_prob.target),
    ],
)
@pytest.mark.parametrize("ignore_index", [None])
class TestDiceBinary(MetricTester):
    @pytest.mark.parametrize("ddp", [False])
    @pytest.mark.parametrize("dist_sync_on_step", [False])
    def test_dice_class(self, ddp, dist_sync_on_step, preds, target, ignore_index):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=Dice,
            sk_metric=partial(_sk_dice, ignore_index=ignore_index),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={"ignore_index": ignore_index},
        )

    def test_dice_fn(self, preds, target, ignore_index):
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=dice,
            sk_metric=partial(_sk_dice, ignore_index=ignore_index),
            metric_args={"ignore_index": ignore_index},
        )


@pytest.mark.parametrize(
    "preds, target",
    [
        (_input_mcls.preds, _input_mcls.target),
        (_input_mcls_logits.preds, _input_mcls_logits.target),
        (_input_mcls_prob.preds, _input_mcls_prob.target),
        (_input_miss_class.preds, _input_miss_class.target),
        (_input_mlb.preds, _input_mlb.target),
        (_input_mlb_logits.preds, _input_mlb_logits.target),
        (_input_mlmd.preds, _input_mlmd.target),
        (_input_mlmd_prob.preds, _input_mlmd_prob.target),
        (_input_mlb_prob.preds, _input_mlb_prob.target),
    ],
)
@pytest.mark.parametrize("ignore_index", [None, 0])
class TestDiceMulti(MetricTester):
    @pytest.mark.parametrize("ddp", [False])
    @pytest.mark.parametrize("dist_sync_on_step", [False])
    def test_dice_class(self, ddp, dist_sync_on_step, preds, target, ignore_index):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=Dice,
            sk_metric=partial(_sk_dice, ignore_index=ignore_index),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={"ignore_index": ignore_index},
        )

    def test_dice_fn(self, preds, target, ignore_index):
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=dice,
            sk_metric=partial(_sk_dice, ignore_index=ignore_index),
            metric_args={"ignore_index": ignore_index},
        )
