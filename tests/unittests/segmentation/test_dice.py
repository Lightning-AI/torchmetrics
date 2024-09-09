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

import pytest
import torch

from monai.metrics.meandice import compute_dice
from torchmetrics.functional.segmentation.dice import dice_score
from torchmetrics.segmentation.dice import DiceScore
from sklearn.metrics import jaccard_score

from unittests import BATCH_SIZE, NUM_BATCHES, NUM_CLASSES, _Input
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

seed_all(42)

_inputs1 = _Input(
    preds=torch.randint(0, 2, (NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, 16)),
    target=torch.randint(0, 2, (NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, 16)),
)
_inputs2 = _Input(
    preds=torch.randint(0, 2, (NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, 32, 32)),
    target=torch.randint(0, 2, (NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, 32, 32)),
)
_inputs3 = _Input(
    preds=torch.randint(0, NUM_CLASSES, (NUM_BATCHES, BATCH_SIZE, 32, 32)),
    target=torch.randint(0, NUM_CLASSES, (NUM_BATCHES, BATCH_SIZE, 32, 32)),
)

def sklearn_dice(*args, **kwargs):
    js = jaccard_score(*args, **kwargs)
    return 2 * js / (1 + js)


def _reference_dice_score(
    preds: torch.Tensor,
    target: torch.Tensor,
    input_format: str,
    include_background: bool = True,
    reduce: bool = True,
):
    pass


@pytest.mark.parametrize(
    "preds, target, input_format",
    [
        (_inputs1.preds, _inputs1.target, "one-hot"),
        (_inputs2.preds, _inputs2.target, "one-hot"),
        (_inputs3.preds, _inputs3.target, "index"),
    ],
)
@pytest.mark.parametrize("include_background", [True, False])
class TestGeneralizedDiceScore(MetricTester):
    """Test class for `DiceScore` metric."""