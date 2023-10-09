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
from typing import Any

import pytest
import torch
from torch import Tensor

from unittests import BATCH_SIZE, EXTRA_DIM, NUM_BATCHES, NUM_CLASSES, _GroupInput, _Input
from unittests.helpers import seed_all

seed_all(1)


def _inv_sigmoid(x: Tensor) -> Tensor:
    return (x / (1 - x)).log()


def _logsoftmax(x: Tensor, dim: int = -1) -> Tensor:
    return torch.nn.functional.log_softmax(x, dim)


_input_binary_prob = _Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE), target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE))
)

_input_binary = _Input(
    preds=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE)),
    target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE)),
)

_input_binary_logits = _Input(
    preds=torch.randn(NUM_BATCHES, BATCH_SIZE), target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE))
)

_input_multilabel_prob = _Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES),
    target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES)),
)

_input_multilabel_multidim_prob = _Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, EXTRA_DIM),
    target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, EXTRA_DIM)),
)

_input_multilabel_logits = _Input(
    preds=torch.randn(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES),
    target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES)),
)

_input_multilabel = _Input(
    preds=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES)),
    target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES)),
)

_input_multilabel_multidim = _Input(
    preds=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, EXTRA_DIM)),
    target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, EXTRA_DIM)),
)

_binary_cases = (
    pytest.param(
        _Input(
            preds=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE)),
            target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE)),
        ),
        id="input[single_dim-labels]",
    ),
    pytest.param(
        _Input(preds=torch.rand(NUM_BATCHES, BATCH_SIZE), target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE))),
        id="input[single_dim-probs]",
    ),
    pytest.param(
        _Input(
            preds=_inv_sigmoid(torch.rand(NUM_BATCHES, BATCH_SIZE)),
            target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE)),
        ),
        id="input[single_dim-logits]",
    ),
    pytest.param(
        _Input(
            preds=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)),
            target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)),
        ),
        id="input[multi_dim-labels]",
    ),
    pytest.param(
        _Input(
            preds=torch.rand(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM),
            target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)),
        ),
        id="input[multi_dim-probs]",
    ),
    pytest.param(
        _Input(
            preds=_inv_sigmoid(torch.rand(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)),
            target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)),
        ),
        id="input[multi_dim-logits]",
    ),
)


def _multiclass_with_missing_class(*shape: Any, num_classes=NUM_CLASSES):
    """Generate multiclass input where a class is missing.

    Args:
        shape: shape of the tensor
        num_classes: number of classes

    Returns:
        tensor with missing classes

    """
    x = torch.randint(0, num_classes, shape)
    x[x == 0] = 2
    return x


_multiclass_cases = (
    pytest.param(
        _Input(
            preds=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE)),
            target=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE)),
        ),
        id="input[single_dim-labels]",
    ),
    pytest.param(
        _Input(
            preds=torch.randn(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES).softmax(-1),
            target=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE)),
        ),
        id="input[single_dim-probs]",
    ),
    pytest.param(
        _Input(
            preds=_logsoftmax(torch.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES), -1),
            target=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE)),
        ),
        id="input[single_dim-logits]",
    ),
    pytest.param(
        _Input(
            preds=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)),
            target=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)),
        ),
        id="input[multi_dim-labels]",
    ),
    pytest.param(
        _Input(
            preds=torch.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, EXTRA_DIM).softmax(-2),
            target=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)),
        ),
        id="input[multi_dim-probs]",
    ),
    pytest.param(
        _Input(
            preds=_logsoftmax(torch.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, EXTRA_DIM), -2),
            target=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)),
        ),
        id="input[multi_dim-logits]",
    ),
    pytest.param(
        _Input(
            preds=_multiclass_with_missing_class(NUM_BATCHES, BATCH_SIZE, num_classes=NUM_CLASSES),
            target=_multiclass_with_missing_class(NUM_BATCHES, BATCH_SIZE, num_classes=NUM_CLASSES),
        ),
        id="input[single_dim-labels-missing_class]",
    ),
)


_multilabel_cases = (
    pytest.param(
        _Input(
            preds=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES)),
            target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES)),
        ),
        id="input[single_dim-labels]",
    ),
    pytest.param(
        _Input(
            preds=torch.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES),
            target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES)),
        ),
        id="input[single_dim-probs]",
    ),
    pytest.param(
        _Input(
            preds=_inv_sigmoid(torch.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES)),
            target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES)),
        ),
        id="input[single_dim-logits]",
    ),
    pytest.param(
        _Input(
            preds=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, EXTRA_DIM)),
            target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, EXTRA_DIM)),
        ),
        id="input[multi_dim-labels]",
    ),
    pytest.param(
        _Input(
            preds=torch.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, EXTRA_DIM),
            target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, EXTRA_DIM)),
        ),
        id="input[multi_dim-probs]",
    ),
    pytest.param(
        _Input(
            preds=_inv_sigmoid(torch.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, EXTRA_DIM)),
            target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, EXTRA_DIM)),
        ),
        id="input[multi_dim-logits]",
    ),
)


_group_cases = (
    pytest.param(
        _GroupInput(
            preds=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE)),
            target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE)),
            groups=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE)),
        ),
        id="input[single_dim-labels]",
    ),
    pytest.param(
        _GroupInput(
            preds=torch.rand(NUM_BATCHES, BATCH_SIZE),
            target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE)),
            groups=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE)),
        ),
        id="input[single_dim-probs]",
    ),
    pytest.param(
        _GroupInput(
            preds=_inv_sigmoid(torch.rand(NUM_BATCHES, BATCH_SIZE)),
            target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE)),
            groups=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE)),
        ),
        id="input[single_dim-logits]",
    ),
)

# Generate edge multilabel edge case, where nothing matches (scores are undefined)
__temp_preds = torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES))
__temp_target = abs(__temp_preds - 1)

_input_multilabel_no_match = _Input(preds=__temp_preds, target=__temp_target)

__mc_prob_logits = 10 * torch.randn(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES)
__mc_prob_preds = __mc_prob_logits.abs() / __mc_prob_logits.abs().sum(dim=2, keepdim=True)

_input_multiclass_prob = _Input(
    preds=__mc_prob_preds, target=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE))
)

_input_multiclass_logits = _Input(
    preds=__mc_prob_logits, target=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE))
)

_input_multiclass = _Input(
    preds=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE)),
    target=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE)),
)

__mdmc_prob_preds = torch.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, EXTRA_DIM)
__mdmc_prob_preds = __mdmc_prob_preds / __mdmc_prob_preds.sum(dim=2, keepdim=True)

_input_multidim_multiclass_prob = _Input(
    preds=__mdmc_prob_preds, target=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM))
)

_input_multidim_multiclass = _Input(
    preds=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)),
    target=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)),
)


# Generate plausible-looking inputs
def _generate_plausible_inputs_multilabel(num_classes=NUM_CLASSES, num_batches=NUM_BATCHES, batch_size=BATCH_SIZE):
    correct_targets = torch.randint(high=num_classes, size=(num_batches, batch_size))
    preds = torch.rand(num_batches, batch_size, num_classes)
    targets = torch.zeros_like(preds, dtype=torch.long)
    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            targets[i, j, correct_targets[i, j]] = 1
    preds += torch.rand(num_batches, batch_size, num_classes) * targets / 3

    preds = preds / preds.sum(dim=2, keepdim=True)

    return _Input(preds=preds, target=targets)


def _generate_plausible_inputs_binary(num_batches=NUM_BATCHES, batch_size=BATCH_SIZE):
    targets = torch.randint(high=2, size=(num_batches, batch_size))
    preds = torch.rand(num_batches, batch_size) + torch.rand(num_batches, batch_size) * targets / 3
    return _Input(preds=preds / (preds.max() + 0.01), target=targets)


_input_multilabel_prob_plausible = _generate_plausible_inputs_multilabel()

_input_binary_prob_plausible = _generate_plausible_inputs_binary()

# randomly remove one class from the input
_temp = torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE))
_class_remove, _class_replace = torch.multinomial(torch.ones(NUM_CLASSES), num_samples=2, replacement=False)
_temp[_temp == _class_remove] = _class_replace

_input_multiclass_with_missing_class = _Input(_temp.clone(), _temp.clone())


_negmetric_noneavg = {
    "pred1": torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    "target1": torch.tensor([0, 1]),
    "res1": torch.tensor([0.0, 0.0, float("nan")]),
    "pred2": torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    "target2": torch.tensor([0, 2]),
    "res2": torch.tensor([0.0, 0.0, 0.0]),
}
