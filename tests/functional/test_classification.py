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
import torch
from torch import Tensor, tensor

from tests.helpers import seed_all
from torchmetrics.functional import dice_score
from torchmetrics.functional.classification.precision_recall_curve import _binary_clf_curve
from torchmetrics.utilities.data import get_num_classes, to_categorical, to_onehot


def test_onehot():
    test_tensor = tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    expected = torch.stack(
        [
            torch.cat([torch.eye(5, dtype=int), torch.zeros((5, 5), dtype=int)]),
            torch.cat([torch.zeros((5, 5), dtype=int), torch.eye(5, dtype=int)]),
        ]
    )

    assert test_tensor.shape == (2, 5)
    assert expected.shape == (2, 10, 5)

    onehot_classes = to_onehot(test_tensor, num_classes=10)
    onehot_no_classes = to_onehot(test_tensor)

    assert torch.allclose(onehot_classes, onehot_no_classes)

    assert onehot_classes.shape == expected.shape
    assert onehot_no_classes.shape == expected.shape

    assert torch.allclose(expected.to(onehot_no_classes), onehot_no_classes)
    assert torch.allclose(expected.to(onehot_classes), onehot_classes)


def test_to_categorical():
    test_tensor = torch.stack(
        [
            torch.cat([torch.eye(5, dtype=int), torch.zeros((5, 5), dtype=int)]),
            torch.cat([torch.zeros((5, 5), dtype=int), torch.eye(5, dtype=int)]),
        ]
    ).to(torch.float)

    expected = tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    assert expected.shape == (2, 5)
    assert test_tensor.shape == (2, 10, 5)

    result = to_categorical(test_tensor)

    assert result.shape == expected.shape
    assert torch.allclose(result, expected.to(result.dtype))


@pytest.mark.parametrize(
    ["preds", "target", "num_classes", "expected_num_classes"],
    [
        pytest.param(torch.rand(32, 10, 28, 28), torch.randint(10, (32, 28, 28)), 10, 10),
        pytest.param(torch.rand(32, 10, 28, 28), torch.randint(10, (32, 28, 28)), None, 10),
        pytest.param(torch.rand(32, 28, 28), torch.randint(10, (32, 28, 28)), None, 10),
    ],
)
def test_get_num_classes(preds, target, num_classes, expected_num_classes):
    assert get_num_classes(preds, target, num_classes) == expected_num_classes


@pytest.mark.parametrize(
    ["sample_weight", "pos_label", "exp_shape"],
    [
        pytest.param(1, 1.0, 42),
        pytest.param(None, 1.0, 42),
    ],
)
def test_binary_clf_curve(sample_weight, pos_label, exp_shape):
    # TODO: move back the pred and target to test func arguments
    #  if you fix the array inside the function, you'd also have fix the shape,
    #  because when the array changes, you also have to fix the shape
    seed_all(0)
    pred = torch.randint(low=51, high=99, size=(100,), dtype=torch.float) / 100
    target = tensor([0, 1] * 50, dtype=torch.int)
    if sample_weight is not None:
        sample_weight = torch.ones_like(pred) * sample_weight

    fps, tps, thresh = _binary_clf_curve(preds=pred, target=target, sample_weights=sample_weight, pos_label=pos_label)

    assert isinstance(tps, Tensor)
    assert isinstance(fps, Tensor)
    assert isinstance(thresh, Tensor)
    assert tps.shape == (exp_shape,)
    assert fps.shape == (exp_shape,)
    assert thresh.shape == (exp_shape,)


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
