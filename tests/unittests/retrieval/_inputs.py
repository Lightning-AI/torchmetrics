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
from typing import NamedTuple

import torch
from torch import Tensor

from unittests import BATCH_SIZE, EXTRA_DIM, NUM_BATCHES


class _Input(NamedTuple):
    indexes: Tensor
    preds: Tensor
    target: Tensor


# correct
_input_retrieval_scores = _Input(
    indexes=torch.randint(high=10, size=(NUM_BATCHES, BATCH_SIZE)),
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE),
    target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE)),
)

_input_retrieval_scores_for_adaptive_k = _Input(
    indexes=torch.randint(high=NUM_BATCHES * BATCH_SIZE // 2, size=(NUM_BATCHES, BATCH_SIZE)),
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE),
    target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE)),
)

_input_retrieval_scores_extra = _Input(
    indexes=torch.randint(high=10, size=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)),
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM),
    target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)),
)

_input_retrieval_scores_int_target = _Input(
    indexes=torch.randint(high=10, size=(NUM_BATCHES, 2 * BATCH_SIZE)),
    preds=torch.rand(NUM_BATCHES, 2 * BATCH_SIZE),
    target=torch.randint(low=-1, high=4, size=(NUM_BATCHES, 2 * BATCH_SIZE)),
)

_input_retrieval_scores_float_target = _Input(
    indexes=torch.randint(high=10, size=(NUM_BATCHES, 2 * BATCH_SIZE)),
    preds=torch.rand(NUM_BATCHES, 2 * BATCH_SIZE),
    target=torch.rand(NUM_BATCHES, 2 * BATCH_SIZE),
)

_input_retrieval_scores_with_ignore_index = _Input(
    indexes=torch.randint(high=10, size=(NUM_BATCHES, BATCH_SIZE)),
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE),
    target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE)).masked_fill(
        mask=torch.randn(NUM_BATCHES, BATCH_SIZE) > 0.5, value=-100
    ),
)

# with errors
_input_retrieval_scores_no_target = _Input(
    indexes=torch.randint(high=10, size=(NUM_BATCHES, BATCH_SIZE)),
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE),
    target=torch.randint(high=1, size=(NUM_BATCHES, BATCH_SIZE)),
)

_input_retrieval_scores_all_target = _Input(
    indexes=torch.randint(high=10, size=(NUM_BATCHES, BATCH_SIZE)),
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE),
    target=torch.randint(low=1, high=2, size=(NUM_BATCHES, BATCH_SIZE)),
)

_input_retrieval_scores_empty = _Input(
    indexes=torch.randint(high=10, size=[0]),
    preds=torch.rand(0),
    target=torch.randint(high=2, size=[0]),
)

_input_retrieval_scores_mismatching_sizes = _Input(
    indexes=torch.randint(high=10, size=(NUM_BATCHES, BATCH_SIZE - 2)),
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE),
    target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE)),
)

_input_retrieval_scores_mismatching_sizes_func = _Input(
    indexes=torch.randint(high=10, size=(NUM_BATCHES, BATCH_SIZE)),
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE - 2),
    target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE)),
)

_input_retrieval_scores_wrong_targets = _Input(
    indexes=torch.randint(high=10, size=(NUM_BATCHES, BATCH_SIZE)),
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE),
    target=torch.randint(low=-(2**31), high=2**31, size=(NUM_BATCHES, BATCH_SIZE)),
)
