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

import torch

from tests.helpers.testers import BATCH_SIZE, EXTRA_DIM, NUM_BATCHES

Input = namedtuple('InputMultiple', ["indexes", "preds", "target"])

# correct
_input_retrieval_scores = Input(
    indexes=torch.randint(high=10, size=(NUM_BATCHES, BATCH_SIZE)),
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE),
    target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE)),
)

_input_retrieval_scores_extra = Input(
    indexes=torch.randint(high=10, size=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)),
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM),
    target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)),
)

# with errors
_input_retrieval_scores_no_target = Input(
    indexes=torch.randint(high=10, size=(NUM_BATCHES, BATCH_SIZE)),
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE),
    target=torch.randint(high=1, size=(NUM_BATCHES, BATCH_SIZE)),
)

_input_retrieval_scores_all_target = Input(
    indexes=torch.randint(high=10, size=(NUM_BATCHES, BATCH_SIZE)),
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE),
    target=torch.randint(low=1, high=2, size=(NUM_BATCHES, BATCH_SIZE)),
)

_input_retrieval_scores_empty = Input(
    indexes=torch.randint(high=10, size=[0]),
    preds=torch.rand(0),
    target=torch.randint(high=2, size=[0]),
)

_input_retrieval_scores_mismatching_sizes = Input(
    indexes=torch.randint(high=10, size=(NUM_BATCHES, BATCH_SIZE - 2)),
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE),
    target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE)),
)

_input_retrieval_scores_mismatching_sizes_func = Input(
    indexes=torch.randint(high=10, size=(NUM_BATCHES, BATCH_SIZE)),
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE - 2),
    target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE)),
)

_input_retrieval_scores_wrong_targets = Input(
    indexes=torch.randint(high=10, size=(NUM_BATCHES, BATCH_SIZE)),
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE),
    target=torch.randint(low=-2**31, high=2**31, size=(NUM_BATCHES, BATCH_SIZE)),
)
