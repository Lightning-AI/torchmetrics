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
from torch import tensor

from torchmetrics.functional import dice_score


@pytest.mark.parametrize(
    ["pred", "target", "expected"],
    [
        pytest.param([[0, 0], [1, 1]], [[0, 0], [1, 1]], 1.0),
        pytest.param([[1, 1], [0, 0]], [[0, 0], [1, 1]], 0.0),
        pytest.param([[1, 1], [1, 1]], [[1, 1], [0, 0]], 2 / 3),
        pytest.param([[1, 1], [0, 0]], [[1, 1], [0, 0]], 1.0),
    ],
)
def test_dice_score(pred, target, expected):
    score = dice_score(tensor(pred), tensor(target))
    assert score == expected
