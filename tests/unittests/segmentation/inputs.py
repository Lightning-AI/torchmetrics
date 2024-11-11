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
__all__ = ["_Input"]

import torch

from unittests import BATCH_SIZE, NUM_BATCHES, NUM_CLASSES, _Input
from unittests._helpers import seed_all

seed_all(42)

to_one_hot = lambda x: torch.nn.functional.one_hot(x, NUM_CLASSES).permute(0, 1, 4, 2, 3)

_inputs1 = _Input(
    preds=to_one_hot(torch.randint(0, NUM_CLASSES, (NUM_BATCHES, BATCH_SIZE, 16, 16))),
    target=to_one_hot(torch.randint(0, NUM_CLASSES, (NUM_BATCHES, BATCH_SIZE, 16, 16))),
)
_inputs2 = _Input(
    preds=to_one_hot(torch.randint(0, NUM_CLASSES, (NUM_BATCHES, BATCH_SIZE, 32, 32))),
    target=to_one_hot(torch.randint(0, NUM_CLASSES, (NUM_BATCHES, BATCH_SIZE, 32, 32))),
)
_inputs3 = _Input(
    preds=torch.randint(0, NUM_CLASSES, (NUM_BATCHES, BATCH_SIZE, 32, 32)),
    target=torch.randint(0, NUM_CLASSES, (NUM_BATCHES, BATCH_SIZE, 32, 32)),
)
_input4 = _Input(
    preds=torch.randint(0, NUM_CLASSES, (NUM_BATCHES, BATCH_SIZE, 32)),
    target=torch.randint(0, NUM_CLASSES, (NUM_BATCHES, BATCH_SIZE, 32)),
)
